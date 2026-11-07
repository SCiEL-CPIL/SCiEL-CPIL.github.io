"""Neural ODE regression: learns dy/dt = f(x, t, theta) to model y as a trajectory."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def _mlp_forward_vec(X_in, W, b):
    """Vectorized forward pass through MLP with tanh activations."""
    z = X_in
    for i in range(len(W) - 1):
        z = np.tanh(z @ W[i] + b[i])
    return z @ W[-1] + b[-1]


def _init_mlp(input_dim, hidden_dims, output_dim, seed=0):
    """Initialize MLP weights with Xavier uniform."""
    rng = np.random.default_rng(seed)
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    W, b = [], []
    for i in range(len(dims) - 1):
        scale = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        W.append(rng.uniform(-scale, scale, size=(dims[i], dims[i + 1])))
        b.append(np.zeros(dims[i + 1]))
    return W, b


def _rk4_integrate_vectorized(x_vals, W, b, n_steps=30, t_max=1.0):
    """
    Vectorized RK4 integration of dy/dt = f(x, t, theta) for all x simultaneously.
    Returns y(t_max) for each x.
    """
    n = len(x_vals)
    y = np.zeros(n)
    dt = t_max / n_steps
    
    # Precompute time points
    t_points = np.arange(n_steps) * dt
    
    for step in range(n_steps):
        t = step * dt
        t_norm = t / t_max
        
        # Build input matrix: (n, 2) -> [x, t_norm]
        inp = np.column_stack([x_vals, np.full(n, t_norm)])
        
        # RK4: k1, k2, k3, k4
        k1 = _mlp_forward_vec(inp, W, b).flatten()
        
        inp2 = np.column_stack([x_vals, np.full(n, (t + dt/2) / t_max)])
        y2 = y + dt/2 * k1
        # Note: our ODE is dy/dt = f(x,t), independent of y, so y doesn't affect f
        # But for generality, we include y in state if needed
        # Here f depends only on x and t
        k2 = _mlp_forward_vec(inp2, W, b).flatten()
        
        inp3 = np.column_stack([x_vals, np.full(n, (t + dt/2) / t_max)])
        k3 = _mlp_forward_vec(inp3, W, b).flatten()
        
        inp4 = np.column_stack([x_vals, np.full(n, (t + dt) / t_max)])
        k4 = _mlp_forward_vec(inp4, W, b).flatten()
        
        y = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    return y


def _forward_with_activations(X_in, W, b):
    """Forward pass returning all layer activations for backprop."""
    acts = [X_in]
    z = X_in
    for i in range(len(W) - 1):
        z = np.tanh(z @ W[i] + b[i])
        acts.append(z)
    z = z @ W[-1] + b[-1]
    acts.append(z)
    return acts


def _backward(acts, dL_dout, W, b):
    """Backprop through MLP."""
    n_layers = len(W)
    dW = [None] * n_layers
    db = [None] * n_layers
    dz = dL_dout.copy()
    
    for i in range(n_layers - 1, -1, -1):
        if i < n_layers - 1:
            dz = dz * (1 - acts[i+1] ** 2)
        dW[i] = acts[i].T @ dz
        db[i] = dz.sum(axis=0)
        if i > 0:
            dz = dz @ W[i].T
    
    return dW, db


def _train_neural_ode(X_train, y_train, max_iter=1500, lr=0.01, seed=42, hidden_dims=(64, 64)):
    """
    Train a Neural ODE: dy/dt = f(x, t, theta) where f is an MLP.
    y(x) = integral_0^1 f(x, t, theta) dt
    Trained via backprop through the RK4 discretization.
    """
    x = X_train[:, 0] if X_train.ndim > 1 else X_train
    y = y_train
    n = len(x)
    
    # Normalize x to [-1, 1]
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    x_norm = (x - x_min) / x_range * 2 - 1
    
    # Normalize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    t_max = 1.0
    W, b = _init_mlp(2, list(hidden_dims), 1, seed=seed)
    
    # Number of RK4 steps for training (coarser for speed)
    n_steps_train = 20
    dt = t_max / n_steps_train
    
    # Precompute time grid
    t_grid = np.arange(n_steps_train + 1) * dt
    
    # We'll use a trapezoidal rule approximation for the integral
    # and backprop through the MLP at each time step
    
    best_loss = np.inf
    best_W = [w.copy() for w in W]
    best_b = [bb.copy() for bb in b]
    
    for it in range(max_iter):
        # Forward: compute f(x, t) at each time step
        # Build input: for each (x_i, t_j), input is [x_norm_i, t_j/t_max]
        # Shape: (n * n_steps, 2)
        X_in = np.zeros((n * (n_steps_train + 1), 2))
        X_in[:, 0] = np.tile(x_norm, n_steps_train + 1)
        X_in[:, 1] = np.repeat(t_grid / t_max, n)
        
        # Forward pass
        acts = _forward_with_activations(X_in, W, b)
        F = acts[-1].reshape(n, n_steps_train + 1)  # (n, n_steps+1)
        
        # Trapezoidal integration
        y_pred = dt * (0.5 * F[:, 0] + F[:, 1:-1].sum(axis=1) + 0.5 * F[:, -1])
        
        # Loss
        errors = y_pred - y_norm
        loss = np.mean(errors ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_b = [bb.copy() for bb in b]
        
        if loss < 1e-12:
            break
        
        # Backprop
        dL_dy = 2.0 * errors / n  # (n,)
        # dL/dF at each time step (trapezoidal weights)
        dL_dF = np.zeros_like(F)
        dL_dF[:, 0] = dL_dy * dt * 0.5
        dL_dF[:, -1] = dL_dy * dt * 0.5
        for j in range(1, n_steps_train):
            dL_dF[:, j] = dL_dy * dt
        
        dL_dout = dL_dF.reshape(-1, 1)
        dW, db = _backward(acts, dL_dout, W, b)
        
        # Update with Adam-like momentum
        for i in range(len(W)):
            W[i] -= lr * (dW[i] + 0.001 * W[i])
            b[i] -= lr * db[i]
    
    W, b = best_W, best_b
    
    # Prediction function using finer RK4
    def predict(X):
        x_pred = X[:, 0] if X.ndim > 1 else X
        x_pred_norm = (x_pred - x_min) / x_range * 2 - 1
        y_pred_norm = _rk4_integrate_vectorized(x_pred_norm, W, b, n_steps=50, t_max=t_max)
        return y_pred_norm * y_std + y_mean
    
    return predict, best_loss


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
    
    best_predict = None
    best_score = np.inf
    
    # Try multiple configurations
    configs = [
        (1500, 0.01, 42, (64, 64)),
        (1500, 0.005, 123, (64, 64)),
        (1500, 0.01, 777, (48, 48)),
        (1000, 0.02, 999, (80, 80)),
    ]
    
    for max_iter, lr, seed, hidden in configs:
        try:
            predict, train_loss = _train_neural_ode(
                X_train, y_train, max_iter=max_iter, lr=lr, seed=seed, hidden_dims=hidden
            )
            # Evaluate on training data
            y_pred = predict(X_train)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y_train) ** 2)
                if mse < best_score:
                    best_score = mse
                    best_predict = predict
        except Exception:
            continue
    
    if best_predict is not None:
        return best_predict
    
    # Fallback: polynomial regression
    def features(X):
        x = X[:, 0]
        cols = [np.ones(len(x))]
        for k in range(1, 8):
            cols.append(x ** k)
        return np.column_stack(cols)
    
    try:
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    except Exception:
        c = np.mean(y_train)
        return lambda X: np.full(len(X), c)

# EVOLVE-BLOCK-END
