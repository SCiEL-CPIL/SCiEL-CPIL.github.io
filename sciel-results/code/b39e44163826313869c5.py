"""Neural ODE regression: learns dy/dt = f(y, x, t, theta) to model y as a trajectory."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def _mlp_forward(z, W, b):
    """Forward pass through a 2-hidden-layer MLP with tanh activations."""
    for i in range(len(W) - 1):
        z = np.tanh(z @ W[i] + b[i])
    return z @ W[-1] + b[-1]


def _init_mlp(input_dim, hidden_dims, output_dim, seed=0):
    """Initialize MLP weights with small random values."""
    rng = np.random.default_rng(seed)
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    W, b = [], []
    for i in range(len(dims) - 1):
        scale = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        W.append(rng.uniform(-scale, scale, size=(dims[i], dims[i + 1])))
        b.append(np.zeros(dims[i + 1]))
    return W, b


def _train_neural_ode(X_train, y_train, max_iter=1000, lr=0.01, seed=42, n_steps=20):
    """
    Train a Neural ODE to fit y = F(x) by learning dy/dt = f(y, x, t, theta).
    Uses vectorized RK4 integration and backpropagation through the time steps.
    """
    x = X_train[:, 0] if X_train.ndim > 1 else X_train
    y = y_train
    n = len(x)

    # Normalize inputs to [-1, 1] for stability
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    x_norm = (x - x_min) / x_range * 2 - 1  # to [-1, 1]

    # Normalize outputs to zero-mean, unit-variance for training
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    t_max = 1.0
    hidden_dims = [32, 32]
    # Input to MLP: [y_current, x, t_norm]
    input_dim = 3
    W, b = _init_mlp(input_dim, hidden_dims, 1, seed=seed)

    dt = t_max / n_steps

    def compute_predictions_and_grads(x_vals, W, b):
        """
        Vectorized forward pass through time and backward pass for gradients.
        Returns y_pred and gradients for W, b.
        """
        n_points = len(x_vals)
        # y_state shape: (n_points,)
        y_state = np.zeros(n_points)
        
        # Store activations for backprop
        # We need to store inputs and outputs of each MLP call at each time step
        # For each time step j, we have n_points samples.
        # We store:
        #   X_in: (n_points, 3)
        #   A1: (n_points, 32)
        #   A2: (n_points, 32)
        #   Out: (n_points, 1)
        
        X_ins = []
        A1s = []
        A2s = []
        Outs = []

        # Forward pass
        for j in range(n_steps):
            t_norm = (j * dt) / t_max
            # Construct inputs: [y_state, x_vals, t_norm]
            X_in = np.column_stack([y_state, x_vals, np.full(n_points, t_norm)])
            
            # MLP Forward
            z = X_in
            A1 = np.tanh(z @ W[0] + b[0])
            A2 = np.tanh(A1 @ W[1] + b[1])
            dydt = A2 @ W[2] + b[2] # (n_points, 1)
            dydt = dydt.flatten() # (n_points,)

            # Store for backward
            X_ins.append(X_in)
            A1s.append(A1)
            A2s.append(A2)
            Outs.append(dydt)

            # RK4 Update
            # k1 = f(y, x, t)
            k1 = dydt
            
            # k2 = f(y + 0.5*dt*k1, x, t + 0.5*dt)
            y_mid1 = y_state + 0.5 * dt * k1
            t_norm_mid = ((j + 0.5) * dt) / t_max
            X_in_mid1 = np.column_stack([y_mid1, x_vals, np.full(n_points, t_norm_mid)])
            z = X_in_mid1
            A1_m1 = np.tanh(z @ W[0] + b[0])
            A2_m1 = np.tanh(A1_m1 @ W[1] + b[1])
            k2 = (A2_m1 @ W[2] + b[2]).flatten()
            
            # Store mid activations (needed for gradient)
            # To save memory and complexity, we might recompute mid steps in backward
            # or store them. Let's store them for exact gradient.
            X_ins.append(X_in_mid1)
            A1s.append(A1_m1)
            A2s.append(A2_m1)
            Outs.append(k2)

            # k3 = f(y + 0.5*dt*k2, x, t + 0.5*dt)
            y_mid2 = y_state + 0.5 * dt * k2
            X_in_mid2 = np.column_stack([y_mid2, x_vals, np.full(n_points, t_norm_mid)])
            z = X_in_mid2
            A1_m2 = np.tanh(z @ W[0] + b[0])
            A2_m2 = np.tanh(A1_m2 @ W[1] + b[1])
            k3 = (A2_m2 @ W[2] + b[2]).flatten()
            
            X_ins.append(X_in_mid2)
            A1s.append(A1_m2)
            A2s.append(A2_m2)
            Outs.append(k3)

            # k4 = f(y + dt*k3, x, t + dt)
            y_end = y_state + dt * k3
            t_norm_end = ((j + 1) * dt) / t_max
            X_in_end = np.column_stack([y_end, x_vals, np.full(n_points, t_norm_end)])
            z = X_in_end
            A1_m3 = np.tanh(z @ W[0] + b[0])
            A2_m3 = np.tanh(A1_m3 @ W[1] + b[1])
            k4 = (A2_m3 @ W[2] + b[2]).flatten()
            
            X_ins.append(X_in_end)
            A1s.append(A1_m3)
            A2s.append(A2_m3)
            Outs.append(k4)

            # Update y
            y_state += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        y_pred = y_state

        # Backward pass
        # Loss L = 0.5 * sum((y_pred - y_target)^2)
        # dL/dy_pred = y_pred - y_target
        dy_pred = y_pred - y_norm
        dL_dy_state = dy_pred / n # Gradient w.r.t final state

        dW = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bi) for bi in b]

        # We iterate backwards through time steps
        # The state update is: y_{j+1} = y_j + dt/6 * (k1 + 2k2 + 2k3 + k4)
        # k1, k2, k3, k4 depend on y_j and parameters.
        
        # We need to accumulate gradients for y_state at each step j.
        # Let G_j be the gradient of Loss w.r.t y_j.
        # G_{n_steps} = dL_dy_state
        # G_j = G_{j+1} * (1 + dt/6 * (d k1/dy_j + 2 d k2/dy_j + 2 d k3/dy_j + d k4/dy_j))
        #       + G_{j+1} * dt/6 * (d k1/dy_j + ...) -- wait, chain rule.
        
        # Actually, it's easier to just backprop through the stored activations.
        # We have stored all intermediate states.
        # Let's reverse the loop.
        
        # G_y is gradient w.r.t y_state at the end of step j (which is start of j+1)
        G_y = dL_dy_state
        G_k1 = np.zeros(n_points)
        G_k2 = np.zeros(n_points)
        G_k3 = np.zeros(n_points)
        G_k4 = np.zeros(n_points)
        
        # We need to propagate G_y back to the inputs of the MLPs.
        # The structure is nested.
        # For each step j (from last to first):
        #   y_{j+1} = y_j + dt/6 (k1 + 2k2 + 2k3 + k4)
        #   k1 = f(y_j)
        #   k2 = f(y_j + 0.5 dt k1)
        #   k3 = f(y_j + 0.5 dt k2)
        #   k4 = f(y_j + dt k3)
        
        # We will compute gradients for W and b by accumulating contributions from all steps.
        
        # To make it manageable, we'll recompute the forward pass in reverse? No, we stored it.
        # Let's trace the gradients.
        
        # Indexing:
        # Step j uses:
        #   k1: index 4*j
        #   k2: index 4*j + 1
        #   k3: index 4*j + 2
        #   k4: index 4*j + 3
        
        for j in reversed(range(n_steps)):
            # Gradients w.r.t k1, k2, k3, k4
            # y_{j+1} depends on k1..k4 linearly.
            # dL/dk1 = G_y * dt/6
            # dL/dk2 = G_y * dt/3
            # dL/dk3 = G_y * dt/3
            # dL/dk4 = G_y * dt/6
            # Also, G_y contributes to y_j via the chain rule through the dependencies of k's on y_j.
            
            G_k1 = G_y * dt / 6.0
            G_k2 = G_y * dt / 3.0
            G_k3 = G_y * dt / 3.0
            G_k4 = G_y * dt / 6.0
            
            # Now backprop through MLPs for k4, k3, k2, k1
            # k4 depends on y_end = y_j + dt*k3
            # k3 depends on y_mid2 = y_j + 0.5*dt*k2
            # k2 depends on y_mid1 = y_j + 0.5*dt*k1
            # k1 depends on y_j
            
            # We need to accumulate dL/dy_j from all these paths.
            # Let dL_dy_j_accum = 0
            # dL_dy_j_accum += G_k1 * (d k1/dy_j)
            # dL_dy_j_accum += G_k2 * (d k2/dy_mid1) * (d y_mid1/dy_j) + G_k2 * (d k2/dy_mid1) * (d y_mid1/dk1) * (d k1/dy_j)
            # This is getting complex. 
            # Alternative: Use the stored activations and standard backprop for each MLP call,
            # accumulating gradients into dW, db, and computing dL/d(input) for each call.
            # Then sum up the dL/d(input) contributions to get dL/dy_j.
            
            # Let's do this:
            # For each MLP call in reverse order (k4, k3, k2, k1):
            # 1. Compute dL/dW, dL/db, dL/dInput for that call.
            # 2. Add to global dW, db.
            # 3. Accumulate dL/dInput into the appropriate y variable.
            
            # k4 call
            idx = 4*j + 3
            dL_dOut = G_k4
            # Backprop through MLP
            dL_dA2 = dL_dOut @ W[2].T
            dL_dW2 = A2s[idx].T @ dL_dOut
            dL_db2 = dL_dOut.sum(axis=0)
            dL_dA1 = (dL_dA2 @ W[1].T) * (1 - A1s[idx]**2)
            dL_dW1 = A1s[idx].T @ dL_dA1
            dL_db1 = dL_dA1.sum(axis=0)
            dL_dX_in = (dL_dA1 @ W[0].T) * (1 - X_ins[idx][:, 0]**2) # Wait, X_in is input to tanh? No.
            # Correction: 
            # z = X_in
            # A1 = tanh(z @ W0 + b0)
            # dL/dz = (dL_dA1 @ W0.T) * (1 - A1^2)
            # dL/dX_in = dL/dz (since z = X_in)
            
            dL_dz = (dL_dA1 @ W[0].T) * (1 - A1s[idx]**2)
            
            dW[2] += dL_dW2
            db[2] += dL_db2
            dW[1] += dL_dW1
            db[1] += dL_db1
            dW[0] += dL_dz @ X_ins[idx].T
            db[0] += dL_dz.sum(axis=0)
            
            # dL/dX_in is dL_dz.
            # X_in = [y_end, x, t]
            # dL/dy_end = dL_dz[:, 0]
            # dL/dx = dL_dz[:, 1] (ignore, x is constant)
            
            G_y_end = dL_dz[:, 0]
            
            # y_end = y_j + dt * k3
            # dL/dy_j += G_y_end
            # dL/dk3 += G_y_end * dt
            
            G_k3 += G_y_end * dt
            G_y += G_y_end
            
            # k3 call
            idx = 4*j + 2
            dL_dOut = G_k3
            dL_dA2 = dL_dOut @ W[2].T
            dL_dW2 = A2s[idx].T @ dL_dOut
            dL_db2 = dL_dOut.sum(axis=0)
            dL_dA1 = (dL_dA2 @ W[1].T) * (1 - A1s[idx]**2)
            dL_dW1 = A1s[idx].T @ dL_dA1
            dL_db1 = dL_dA1.sum(axis=0)
            dL_dz = (dL_dA1 @ W[0].T) * (1 - A1s[idx]**2)
            
            dW[2] += dL_dW2
            db[2] += dL_db2
            dW[1] += dL_dW1
            db[1] += dL_db1
            dW[0] += dL_dz @ X_ins[idx].T
            db[0] += dL_dz.sum(axis=0)
            
            G_y_mid2 = dL_dz[:, 0]
            
            # y_mid2 = y_j + 0.5 * dt * k2
            G_k2 += G_y_mid2 * 0.5 * dt
            G_y += G_y_mid2
            
            # k2 call
            idx = 4*j + 1
            dL_dOut = G_k2
            dL_dA2 = dL_dOut @ W[2].T
            dL_dW2 = A2s[idx].T @ dL_dOut
            dL_db2 = dL_dOut.sum(axis=0)
            dL_dA1 = (dL_dA2 @ W[1].T) * (1 - A1s[idx]**2)
            dL_dW1 = A1s[idx].T @ dL_dA1
            dL_db1 = dL_dA1.sum(axis=0)
            dL_dz = (dL_dA1 @ W[0].T) * (1 - A1s[idx]**2)
            
            dW[2] += dL_dW2
            db[2] += dL_db2
            dW[1] += dL_dW1
            db[1] += dL_db1
            dW[0] += dL_dz @ X_ins[idx].T
            db[0] += dL_dz.sum(axis=0)
            
            G_y_mid1 = dL_dz[:, 0]
            
            # y_mid1 = y_j + 0.5 * dt * k1
            G_k1 += G_y_mid1 * 0.5 * dt
            G_y += G_y_mid1
            
            # k1 call
            idx = 4*j
            dL_dOut = G_k1
            dL_dA2 = dL_dOut @ W[2].T
            dL_dW2 = A2s[idx].T @ dL_dOut
            dL_db2 = dL_dOut.sum(axis=0)
            dL_dA1 = (dL_dA2 @ W[1].T) * (1 - A1s[idx]**2)
            dL_dW1 = A1s[idx].T @ dL_dA1
            dL_db1 = dL_dA1.sum(axis=0)
            dL_dz = (dL_dA1 @ W[0].T) * (1 - A1s[idx]**2)
            
            dW[2] += dL_dW2
            db[2] += dL_db2
            dW[1] += dL_dW1
            db[1] += dL_db1
            dW[0] += dL_dz @ X_ins[idx].T
            db[0] += dL_dz.sum(axis=0)
            
            G_y_j = dL_dz[:, 0]
            
            # y_j is the state at the start of this step.
            # G_y accumulates to be the gradient w.r.t y_j for the next iteration (j-1)
            # Note: G_y currently holds the gradient from future steps (j+1..end) w.r.t y_{j+1}.
            # We added G_y_end, G_y_mid2, etc. to G_y in the steps above?
            # No, we used G_y to compute G_k's.
            # We need to update G_y to be the gradient w.r.t y_j.
            # G_y (new) = G_y (old, from future) * (dy_{j+1}/dy_j) + dL/dy_j (direct)
            # Wait, my previous logic was:
            # G_y is dL/dy_{j+1}.
            # We computed G_k1..G_k4 using G_y.
            # Then we backpropped k4, k3, k2, k1.
            # In each backprop, we added to G_y the term dL/dy_j from that path.
            # So after processing k1, G_y contains the total dL/dy_j.
            
            # So G_y is now ready for the next step (j-1) where it will be dL/dy_j.
            # But we also need to account for the fact that y_{j+1} depends on y_j directly?
            # No, y_{j+1} = y_j + ... so dL/dy_j includes the direct term from the update equation?
            # The update equation is y_{j+1} = y_j + h(y_j).
            # dL/dy_j = dL/dy_{j+1} * (1 + h'(y_j)) + dL/dy_j (direct from k's?)
            # My manual accumulation:
            # G_k1..4 derived from G_y (which is dL/dy_{j+1}).
            # Backprop through k4 gave dL/dy_end. Added to G_y.
            # Backprop through k3 gave dL/dy_mid2. Added to G_y.
            # ...
            # Backprop through k1 gave dL/dy_j (from k1's input). Added to G_y.
            # But what about the "1" in y_{j+1} = 1*y_j + ...?
            # The chain rule: dL/dy_j = dL/dy_{j+1} * d(y_{j+1})/dy_j.
            # d(y_{j+1})/dy_j = 1 + dt/6 * (d k1/dy_j + 2 d k2/dy_j + ...).
            # The terms d k_i/dy_j are exactly what we computed and added to G_y via the backprop of k_i inputs.
            # So yes, G_y now correctly holds dL/dy_j.
            
            # Pass to next iteration
            # G_y is already updated.
            
        return y_pred, dW, db

    # Training loop
    best_loss = np.inf
    best_W, best_b = [w.copy() for w in W], [bb.copy() for bb in b]
    
    # Simple Adam optimizer
    m_W = [np.zeros_like(w) for w in W]
    v_W = [np.zeros_like(w) for w in W]
    m_b = [np.zeros_like(bi) for bi in b]
    v_b = [np.zeros_like(bi) for bi in b]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for iteration in range(max_iter):
        y_pred, dW, db = compute_predictions_and_grads(x_norm, W, b)
        
        errors = y_pred - y_norm
        loss = np.mean(errors ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_b = [bb.copy() for bb in b]
            
        if loss < 1e-10:
            break
        
        t += 1
        for i in range(len(W)):
            # Update W
            m_W[i] = beta1 * m_W[i] + (1 - beta1) * dW[i]
            v_W[i] = beta2 * v_W[i] + (1 - beta2) * (dW[i] ** 2)
            m_hat = m_W[i] / (1 - beta1 ** t)
            v_hat = v_W[i] / (1 - beta2 ** t)
            W[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Update b
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (db[i] ** 2)
            m_hat = m_b[i] / (1 - beta1 ** t)
            v_hat = v_b[i] / (1 - beta2 ** t)
            b[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
    W, b = best_W, best_b

    def predict(X):
        x_pred = X[:, 0] if X.ndim > 1 else X
        x_pred_norm = (x_pred - x_min) / x_range * 2 - 1
        # Use the same integration as in training
        y_pred, _, _ = compute_predictions_and_grads(x_pred_norm, W, b)
        return y_pred * y_std + y_mean

    return predict


def fit(X_train, y_train):
    """
    Fit a Neural ODE-based regression model.
    Returns a predict(X) callable.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    n, d = X_train.shape

    # Handle constant y
    if np.std(y_train) < 1e-12:
        c = np.mean(y_train)
        return lambda X: np.full(len(X), c)

    # Try Neural ODE approach
    try:
        predict = _train_neural_ode(X_train, y_train, max_iter=500, lr=0.01, seed=42, n_steps=10)
        # Validate
        y_test_pred = predict(X_train[:10])
        if not np.all(np.isfinite(y_test_pred)):
            raise ValueError("Non-finite predictions")
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
