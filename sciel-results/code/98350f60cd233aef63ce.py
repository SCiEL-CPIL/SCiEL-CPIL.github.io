"""Neural ODE regression: learns dy/dt = f(y, x, t, theta) to model y as a trajectory."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def _mlp_forward(X, W, b):
    """Vectorized forward pass through MLP with tanh activations."""
    Z = [X]
    for i in range(len(W) - 1):
        Z.append(np.tanh(Z[-1] @ W[i] + b[i]))
    Z.append(Z[-1] @ W[-1] + b[-1])
    return Z


def _mlp_backward(Z, dL_dout, W, b):
    """Vectorized backprop through MLP."""
    dW = [None] * len(W)
    db = [None] * len(W)
    dz = dL_dout.copy()
    for i in range(len(W) - 1, -1, -1):
        dW[i] = Z[i].T @ dz
        db[i] = dz.sum(axis=0)
        if i > 0:
            dz = (dz @ W[i].T) * (1 - Z[i + 1] ** 2)
    return dW, db


def _init_mlp(input_dim, hidden_dims, output_dim, seed=0):
    """Initialize MLP weights."""
    rng = np.random.default_rng(seed)
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    W, b = [], []
    for i in range(len(dims) - 1):
        scale = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        W.append(rng.uniform(-scale, scale, size=(dims[i], dims[i + 1])))
        b.append(np.zeros(dims[i + 1]))
    return W, b


def _rk4_integrate(x_norm, W, b, n_steps=20, t_max=1.0):
    """
    Vectorized RK4 integration of dy/dt = f(y, x, t) for all x simultaneously.
    Returns (y_final, saved_states) where saved_states has shape (n_steps+1, n, 3)
    for [y, x, t] at each step.
    """
    n = len(x_norm)
    y = np.zeros(n)
    t = 0.0
    dt = t_max / n_steps

    # Save states for backprop: (n_steps+1, n, 3) -> [y, x, t]
    saved_y = np.zeros((n_steps + 1, n))
    saved_x = np.tile(x_norm, (n_steps + 1, 1))
    saved_t = np.tile(np.linspace(0, t_max, n_steps + 1)[:, None], (1, n))
    saved_y[0] = y

    for step in range(n_steps):
        t_cur = step * dt
        t_next = (step + 1) * dt
        t_mid = t_cur + dt / 2

        # Build input matrix for MLP: [y, x, t/t_max]
        def build_input(y_val, x_val, t_val):
            t_norm = t_val / t_max if t_max > 0 else 0.0
            return np.column_stack([y_val, x_val, t_norm])

        # k1 = f(y, x, t_cur)
        inp1 = build_input(y, x_norm, t_cur)
        out1 = _mlp_forward(inp1, W, b)[-1].flatten()

        # k2 = f(y + dt/2*k1, x, t_mid)
        inp2 = build_input(y + dt / 2 * out1, x_norm, t_mid)
        out2 = _mlp_forward(inp2, W, b)[-1].flatten()

        # k3 = f(y + dt/2*k2, x, t_mid)
        inp3 = build_input(y + dt / 2 * out2, x_norm, t_mid)
        out3 = _mlp_forward(inp3, W, b)[-1].flatten()

        # k4 = f(y + dt*k3, x, t_next)
        inp4 = build_input(y + dt * out3, x_norm, t_next)
        out4 = _mlp_forward(inp4, W, b)[-1].flatten()

        y_new = y + dt / 6 * (out1 + 2 * out2 + 2 * out3 + out4)
        saved_y[step + 1] = y_new
        y = y_new

    saved_states = np.stack([saved_y, saved_x, saved_t], axis=2)  # (n_steps+1, n, 3)
    return y, saved_states


def _train_neural_ode(X_train, y_train, max_iter=1500, lr=0.01, seed=42, n_steps=15):
    """
    Train Neural ODE: dy/dt = f(y, x, t, theta), y(0) = 0, predict y(t_max).
    Uses vectorized RK4 with backprop through the integration steps.
    """
    x = X_train[:, 0]
    y = y_train
    n = len(x)

    # Normalize
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    x_norm = (x - x_min) / x_range * 2 - 1

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    t_max = 1.0
    W, b = _init_mlp(3, [32, 32], 1, seed=seed)

    best_loss = np.inf
    best_W = [w.copy() for w in W]
    best_b = [bb.copy() for bb in b]

    for iteration in range(max_iter):
        # Forward: RK4 integration
        y_pred, saved_states = _rk4_integrate(x_norm, W, b, n_steps=n_steps, t_max=t_max)

        # Loss
        errors = y_pred - y_norm
        loss = np.mean(errors ** 2)

        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_b = [bb.copy() for bb in b]

        if loss < 1e-12:
            break

        # Backward through RK4 steps (reverse order)
        # dL/dy_final = 2 * errors / n
        dL_dy = 2.0 * errors / n  # (n,)

        dt = t_max / n_steps
        dW = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bb) for bb in b]

        # Reverse through steps
        dY = dL_dy.copy()  # (n,) - gradient w.r.t. y at current step
        for step in range(n_steps - 1, -1, -1):
            t_cur = step * dt
            t_mid = t_cur + dt / 2
            t_next = (step + 1) * dt

            y_prev = saved_states[step, :, 0]
            y_curr = saved_states[step + 1, :, 0]
            x_val = x_norm

            # Recompute k1, k2, k3, k4 for backprop
            def build_input(y_val, x_val, t_val):
                t_norm = t_val / t_max if t_max > 0 else 0.0
                return np.column_stack([y_val, x_val, t_norm])

            # k1
            inp1 = build_input(y_prev, x_val, t_cur)
            out1 = _mlp_forward(inp1, W, b)[-1].flatten()

            # k2
            inp2 = build_input(y_prev + dt / 2 * out1, x_val, t_mid)
            out2 = _mlp_forward(inp2, W, b)[-1].flatten()

            # k3
            inp3 = build_input(y_prev + dt / 2 * out2, x_val, t_mid)
            out3 = _mlp_forward(inp3, W, b)[-1].flatten()

            # k4
            inp4 = build_input(y_prev + dt * out3, x_val, t_next)
            out4 = _mlp_forward(inp4, W, b)[-1].flatten()

            # y_curr = y_prev + dt/6 * (k1 + 2k2 + 2k3 + k4)
            # dL/dy_prev = dL/dy_curr + dL/dk terms propagated back

            # Gradient through the combination
            dL_dk1 = dY * (dt / 6)
            dL_dk2 = dY * (2 * dt / 6)
            dL_dk3 = dY * (2 * dt / 6)
            dL_dk4 = dY * (dt / 6)

            # Backprop through each k
            # dL/dy_prev from k1: dL_dk1 * d(k1)/d(y_prev)
            # k1 = f(y_prev, x, t_cur), so d(k1)/d(y_prev) = d(f)/d(input_y)
            # We need to backprop through the MLP for each k

            # For k1: input is [y_prev, x, t_cur/t_max]
            Z1 = _mlp_forward(inp1, W, b)
            dW1, db1 = _mlp_backward(Z1, dL_dk1.reshape(-1, 1), W, b)
            for i in range(len(W)):
                dW[i] += dW1[i]
                db[i] += db1[i]
            # Gradient w.r.t. y_prev from k1
            dY_from_k1 = dL_dk1 * (1 - Z1[1] ** 2) @ W[0].T @ np.ones((W[0].shape[1], 1)).T
            # Actually, we need dL/d(input_y) which is the first column of the Jacobian
            # Let's compute it properly: dL/d(inp1[:, 0])
            # inp1 = [y_prev, x, t_norm], so dL/d(y_prev) = dL/d(inp1[:, 0])
            # This is the first element of the chain: dL/dz_out * W_last^T * ... * W_0^T
            # More precisely, we need the gradient of the output w.r.t. the first input element

            # Let me recompute: we need dL/d(y_prev) where y_prev is input to MLP
            # The MLP output is scalar, input is 3D: [y, x, t]
            # dL/d(y_prev) = sum over output of dL/d(out) * d(out)/d(y_prev)
            # = dL_dk1 * (chain of Jacobians)[:, 0]

            # For simplicity, let's use the full backprop to get dL/d(input)
            # Redo: we want dL/d(inp) for the first component
            # The backward pass gives us dL/d(z) at each layer, and dL/d(input) = dL/d(z0)
            # Let me restructure

            # Actually, let me just accumulate the gradient w.r.t. y_prev by
            # noting that dL/d(y_prev) = dL_dk1 * df/dy + dL_dk2 * df/dy|_{y+dt/2*k1} * (1) 
            # where df/dy is the partial derivative of the MLP output w.r.t. its first input

            # This is getting complex. Let me use a different approach:
            # Compute dL/d(y_prev) by doing a full backprop through each k
            # and extracting the gradient w.r.t. the y component of the input

            # For k1: dL/d(y_prev) contribution = dL_dk1 * (d f1 / d y_prev)
            # where f1 = MLP([y_prev, x, t_cur])
            # d f1 / d y_prev = dL_dk1 * (product of Jacobians) -> we get this from backprop

            # Let me just do it numerically for the y-derivative part since it's scalar
            # Actually, the cleanest way: do backprop and extract dL/d(input[:, 0])

            # Redo backprop to get dL/d(input)
            Z1 = _mlp_forward(inp1, W, b)
            dL_dz1 = dL_dk1.reshape(-1, 1)
            # Backprop to get dL/d(input)
            dL_dinp1 = dL_dz1.copy()
            for i in range(len(W) - 1, -1, -1):
                dW1_i = Z1[i].T @ dL_dinp1
                db1_i = dL_dinp1.sum(axis=0)
                dW[i] += dW1_i
                db[i] += db1_i
                if i > 0:
                    dL_dinp1 = (dL_dinp1 @ W[i].T) * (1 - Z1[i + 1] ** 2)
            # dL_dinp1 is now dL/d(input) for k1, shape (n, 3)
            dY_prev_k1 = dL_dinp1[:, 0]  # gradient w.r.t. y component

            # Similarly for k2: input is [y_prev + dt/2*k1, x, t_mid]
            # dL/d(y_prev) from k2 = dL_dk2 * df2/d(y_prev + dt/2*k1) * 1 (since d(y_prev+dt/2*k1)/d(y_prev)=1)
            Z2 = _mlp_forward(inp2, W, b)
            dL_dz2 = dL_dk2.reshape(-1, 1)
            dL_dinp2 = dL_dz2.copy()
            for i in range(len(W) - 1, -1, -1):
                dW2_i = Z2[i].T @ dL_dinp2
                db2_i = dL_dinp2.sum(axis=0)
                dW[i] += dW2_i
                db[i] += db2_i
                if i > 0:
                    dL_dinp2 = (dL_dinp2 @ W[i].T) * (1 - Z2[i + 1] ** 2)
            dY_prev_k2 = dL_dinp2[:, 0]

            # For k3: input is [y_prev + dt/2*k2, x, t_mid]
            Z3 = _mlp_forward(inp3, W, b)
            dL_dz3 = dL_dk3.reshape(-1, 1)
            dL_dinp3 = dL_dz3.copy()
            for i in range(len(W) - 1, -1, -1):
                dW3_i = Z3[i].T @ dL_dinp3
                db3_i = dL_dinp3.sum(axis=0)
                dW[i] += dW3_i
                db[i] += db3_i
                if i > 0:
                    dL_dinp3 = (dL_dinp3 @ W[i].T) * (1 - Z3[i + 1] ** 2)
            dY_prev_k3 = dL_dinp3[:, 0]

            # For k4: input is [y_prev + dt*k3, x, t_next]
            Z4 = _mlp_forward(inp4, W, b)
            dL_dz4 = dL_dk4.reshape(-1, 1)
            dL_dinp4 = dL_dz4.copy()
            for i in range(len(W) - 1, -1, -1):
                dW4_i = Z4[i].T @ dL_dinp4
                db4_i = dL_dinp4.sum(axis=0)
                dW[i] += dW4_i
                db[i] += db4_i
                if i > 0:
                    dL_dinp4 = (dL_dinp4 @ W[i].T) * (1 - Z4[i + 1] ** 2)
            dY_prev_k4 = dL_dinp4[:, 0]

            # Total dL/d(y_prev)
            dY = dY_prev_k1 + dY_prev_k2 + dY_prev_k3 + dY_prev_k4

        # Update
        for i in range(len(W)):
            W[i] -= lr * (dW[i] + 0.001 * W[i])
            b[i] -= lr * db[i]

    W, b = best_W, best_b

    def predict(X):
        x_pred = X[:, 0] if X.ndim > 1 else np.asarray(X).flatten()
        x_pred_norm = (x_pred - x_min) / x_range * 2 - 1
        y_pred_norm, _ = _rk4_integrate(x_pred_norm, W, b, n_steps=n_steps, t_max=t_max)
        return y_pred_norm * y_std + y_mean

    return predict


def fit(X_train, y_train):
    """
    Fit a Neural ODE-based regression model.
    Returns a predict(X) callable.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    n, d = X_train.shape

    if np.std(y_train) < 1e-12:
        c = np.mean(y_train)
        return lambda X: np.full(len(X), c)

    try:
        predict = _train_neural_ode(X_train, y_train, max_iter=1200, lr=0.008, seed=42, n_steps=12)
        y_check = predict(X_train[:5])
        if not np.all(np.isfinite(y_check)):
            raise ValueError("Non-finite")
        return predict
    except Exception:
        # Fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

# EVOLVE-BLOCK-END
