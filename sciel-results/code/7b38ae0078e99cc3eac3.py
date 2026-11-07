# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline, CubicSpline, PchipInterpolator
from scipy.optimize import differential_evolution, minimize_scalar, minimize

_TOTAL_DEADLINE = time.time() + 318.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _poly_features(X, degree):
    """Polynomial features up to given degree in d=1 (or general d)."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for p in range(1, degree + 1):
                    for q in range(1, degree + 2 - p):
                        if p + q <= degree:
                            cols.append((X[:, i] ** p) * (X[:, j] ** q))
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 8):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
        cols.append(np.exp(-x**2))
    return np.column_stack(cols)


def _mixed_features(X):
    """Combine polynomial, trig, and exp features."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for p in range(1, 7):
            cols.append(x ** p)
        for k in range(1, 6):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-8):
    """Solve least squares with optional ridge regularization."""
    try:
        n, p = F.shape
        A = F.copy()
        # Add small ridge
        A[:, 0] *= (1 + ridge_alpha)  # slightly regularize intercept
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-8):
    """Try fitting a linear model on given features. Returns (predict_fn, rmse) or (None, inf)."""
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _fit_spline_interp(X, y, num_knots=15, method='cubic'):
    """Fit a spline via interpolation on sorted points."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None
        num_knots = min(num_knots, n)
        if num_knots < 5:
            num_knots = 5
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
        
        if method == 'pchip':
            s = PchipInterpolator(knot_x, knot_y)
        else:
            s = make_interp_spline(knot_x, knot_y, k=3)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_gaussian_process(X, y, length_scale=1.0, noise=1e-8):
    """Simple Gaussian Process regression with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_norm = y - y_mu
        
        def kernel(X1, X2):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(Xn, Xn) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except:
            K += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K)
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn)
            preds = (Ks @ alpha).flatten() + y_mu
            return np.nan_to_num(preds, nan=y_mu, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_rbf_fixed(X, y, num_centers=15, num_widths=4):
    """Fit RBF network with fixed centers (from data) and optimized widths."""
    try:
        n, d = X.shape
        if n < num_centers + 4:
            num_centers = max(5, n - 4)
        
        # Use evenly spaced quantiles as centers
        if d == 1:
            centers = np.quantile(X[:, 0], np.linspace(0.05, 0.95, num_centers)).reshape(-1, 1)
        else:
            centers = np.quantile(X, np.linspace(0.05, 0.95, num_centers), axis=0).T
        
        def objective(widths):
            F = _rbf_features(X, centers, widths)
            if not np.all(np.isfinite(F)):
                return 1e12
            try:
                ww, *_ = np.linalg.lstsq(F, y, rcond=None)
                pred = F @ ww
                return np.mean((pred - y)**2)
            except:
                return 1e12
        
        bounds = [(0.01, 3.0)] * num_widths
        result = differential_evolution(objective, bounds, maxiter=30, popsize=8, tol=1e-4, seed=42, polish=False)
        
        if result.fun < 1e10:
            best_widths = result.x
            F = _rbf_features(X, centers, best_widths)
            ww, *_ = np.linalg.lstsq(F, y, rcond=None)
            
            def predict(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                Fp = _rbf_features(Xp, centers, best_widths)
                return Fp @ ww
            return _safe_predict(predict)
    except Exception:
        return None


def _rbf_features(X, centers, widths):
    """Gaussian RBF features."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for c in centers:
        for w in widths:
            dist = np.sqrt(np.sum((X - c) ** 2, axis=1))
            cols.append(np.exp(-0.5 * (dist / w) ** 2))
    return np.column_stack(cols)


def _fit_mlp(X, y, hidden_sizes=(64, 64), lr=0.01, epochs=500, seed=42):
    """Small MLP via Adam optimizer."""
    try:
        d = X.shape[1]
        n = X.shape[0]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma

        np.random.seed(seed)
        layers = [d] + list(hidden_sizes) + [1]
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
        b = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        
        mW = [np.zeros_like(w) for w in W]
        vW = [np.zeros_like(w) for w in W]
        mb = [np.zeros_like(bb) for bb in b]
        vb = [np.zeros_like(bb) for bb in b]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def forward(Xb):
            a = Xb
            acts = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                acts.append(a)
            return acts

        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            t += 1
            acts = forward(Xn)
            pred = acts[-1]
            dz = (pred - yn) / n
            
            grads_W = [None] * len(W)
            grads_b = [None] * len(b)
            for i in range(len(W) - 1, -1, -1):
                if i == len(W) - 1:
                    dz_i = dz
                else:
                    dz_i = dz @ W[i].T * (1 - acts[i+1]**2)
                grads_W[i] = acts[i].T @ dz_i
                grads_b[i] = dz_i.sum(axis=0)
            
            for i in range(len(W)):
                mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                vW[i] = beta2 * vW[i] + (1 - beta2) * grads_W[i]**2
                mW_hat = mW[i] / (1 - beta1**t)
                vW_hat = vW[i] / (1 - beta2**t)
                W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                vb[i] = beta2 * vb[i] + (1 - beta2) * grads_b[i]**2
                mb_hat = mb[i] / (1 - beta1**t)
                vb_hat = vb[i] / (1 - beta2**t)
                b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            a = Xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        return predict
    except Exception:
        return None


def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-6):
    """Kernel Ridge Regression with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        w = np.linalg.solve(K, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            Ks = np.exp(-0.5 * sq_dist)
            preds = Ks @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_neural_ode(X, y, n_steps=50, hidden_dim=8, lr=0.01, epochs=200, seed=42):
    """Neural ODE regression: learn dx/dt = f(x, t) via a small MLP,
    integrating from t=0 to t=1. The input x is the feature (normalized input),
    and we learn a vector field that transforms it to the output.
    For 1-D input, we use the input value as the initial state and learn
    a mapping through continuous dynamics."""
    try:
        n, d = X.shape
        if n < 20:
            return None
        
        # Normalize input and output
        x_mu = X.mean(axis=0)
        x_sigma = X.std(axis=0) + 1e-8
        Xn = (X - x_mu) / x_sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma
        
        # We'll model: h(0) = Xn, dh/dt = f(h, t; theta), h(1) = yn
        # Use a simple architecture: f(h, t) = W1 * [h, t, 1] + b1, then tanh, then W2 * h2 + b2
        # State dimension: d (same as input)
        state_dim = d
        hidden = hidden_dim
        
        np.random.seed(seed)
        # Parameters for the ODE vector field
        # f(h, t) = W2 * tanh(W1 * [h, t, 1] + b1) + b2
        W1 = np.random.randn(hidden, state_dim + 2) * np.sqrt(2.0 / (state_dim + 2))
        b1 = np.zeros(hidden)
        W2 = np.random.randn(state_dim, hidden) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(state_dim)
        
        params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
        
        # Adam optimizer state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t_adam = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        def ode_rhs(h, t_val):
            """Compute dh/dt = f(h, t)."""
            # h: (batch, state_dim), t_val: scalar
            batch = h.shape[0]
            t_col = np.full((batch, 1), t_val)
            ones = np.ones((batch, 1))
            concat = np.concatenate([h, t_col, ones], axis=1)  # (batch, state_dim+2)
            z1 = concat @ W1.T + b1  # (batch, hidden)
            a1 = np.tanh(z1)
            z2 = a1 @ W2.T + b2  # (batch, state_dim)
            return z2
        
        def integrate(h0, n_steps_local):
            """Integrate ODE from t=0 to t=1 using Euler's method."""
            h = h0.copy()
            dt = 1.0 / n_steps_local
            for i in range(n_steps_local):
                t_val = i * dt
                dh = ode_rhs(h, t_val)
                h = h + dt * dh
            return h
        
        def compute_loss_and_grads():
            """Compute loss and gradients via finite differences or manual backprop."""
            # For efficiency, use a small batch or full batch with manual gradient
            h0 = Xn.copy()
            h = h0.copy()
            # Store all intermediate states for backprop
            states = [h.copy()]
            for i in range(n_steps):
                t_val = i * (1.0 / n_steps)
                dh = ode_rhs(h, t_val)
                h = h + (1.0 / n_steps) * dh
                states.append(h.copy())
            
            h_final = states[-1]
            pred = h_final[:, 0] if d == 1 else h_final
            # Loss: MSE between pred and yn
            error = pred - yn  # (n,)
            loss = np.mean(error ** 2)
            
            # Backprop through the ODE (adjoint method simplified)
            # dL/dh_final = 2 * error / n (for d=1, just the first component)
            if d == 1:
                dh_final = np.zeros_like(h_final)
                dh_final[:, 0] = 2.0 * error / n
            else:
                dh_final = 2.0 * error / n
            
            # Backprop through steps
            dW1 = np.zeros_like(W1)
            db1 = np.zeros_like(b1)
            dW2 = np.zeros_like(W2)
            db2 = np.zeros_like(b2)
            
            dh = dh_final
            for i in range(n_steps - 1, -1, -1):
                t_val = i * (1.0 / n_steps)
                h_i = states[i]
                
                # Recompute forward at this step
                batch = h_i.shape[0]
                t_col = np.full((batch, 1), t_val)
                ones = np.ones((batch, 1))
                concat = np.concatenate([h_i, t_col, ones], axis=1)
                z1 = concat @ W1.T + b1
                a1 = np.tanh(z1)
                z2 = a1 @ W2.T + b2
                
                # dL/dz2 = dh
                dz2 = dh
                # dL/da1 = dz2 @ W2
                da1 = dz2 @ W2  # (batch, hidden)
                # dL/dz1 = da1 * (1 - a1^2)
                dz1 = da1 * (1 - a1 ** 2)
                # dL/dconcat = dz1 @ W1.T
                dconcat = dz1 @ W1  # (batch, state_dim+2)
                
                # dL/dh_i from dconcat (first state_dim columns)
                dh_from_concat = dconcat[:, :state_dim]
                
                # dL/dW1 = dconcat.T @ a1... wait, dconcat = dz1 @ W1, so dL/dW1 = dz1.T @ concat
                dW1 += dz1.T @ concat
                db1 += dz1.sum(axis=0)
                
                # dL/dW2 = a1.T @ dz2
                dW2 += a1.T @ dz2
                db2 += dz2.sum(axis=0)
                
                # Chain: dh_i gets contribution from dh_from_concat and from the Euler step
                # h_{i+1} = h_i + dt * f(h_i, t_i)
                # So dh_i = dh_from_concat + (1/dt) * dh_next... 
                # Actually: dh_i = dh_from_concat + (1.0/n_steps) * dh_from_next_step
                # But we're going backwards, so dh is already the gradient w.r.t. h_{i+1}
                # We need to add the direct gradient w.r.t. h_i
                dh = dh_from_concat + (1.0 / n_steps) * dh
                
            # Update params
            t_adam += 1
            all_params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
            all_grads = np.concatenate([dW1.flatten(), db1, dW2.flatten(), db2])
            
            m = beta1 * m + (1 - beta1) * all_grads
            v = beta2 * v + (1 - beta2) * all_grads ** 2
            m_hat = m / (1 - beta1 ** t_adam)
            v_hat = v / (1 - beta2 ** t_adam)
            
            new_params = all_params - lr * m_hat / (np.sqrt(v_hat) + eps)
            W1 = new_params[:hidden*(state_dim+2)].reshape(hidden, state_dim+2)
            b1 = new_params[hidden*(state_dim+2):hidden*(state_dim+2)+hidden]
            W2 = new_params[hidden*(state_dim+2)+hidden:2*hidden*(state_dim+2)+hidden].reshape(state_dim, hidden)
            b2 = new_params[2*hidden*(state_dim+2)+hidden:]
            
            return loss, W1, b1, W2, b2, m, v, t_adam
        
        # Training loop
        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            loss, W1, b1, W2, b2, m, v, t_adam = compute_loss_and_grads()
        
        # Final integration for prediction
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - x_mu) / x_sigma
            h = Xpn.copy()
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t_val = i * dt
                batch = h.shape[0]
                t_col = np.full((batch, 1), t_val)
                ones = np.ones((batch, 1))
                concat = np.concatenate([h, t_col, ones], axis=1)
                z1 = concat @ W1.T + b1
                a1 = np.tanh(z1)
                z2 = a1 @ W2.T + b2
                h = h + dt * z2
            if d == 1:
                pred = h[:, 0]
            else:
                pred = h
            result = pred * y_sigma + y_mu
            return np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return _safe_predict(predict)
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try polynomial degrees
    for deg in [1, 2, 3, 4, 5, 6, 7, 8]:
        if _remaining() < 10.0:
            break
        predict, rmse = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try trig
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_trig_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try exp/log
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_exp_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try mixed
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_mixed_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try splines with different knot counts and methods
    if _remaining() > 10.0 and d == 1:
        for nk in [8, 12, 16, 20, 25]:
            if _remaining() < 5.0:
                break
            for meth in ['cubic', 'pchip']:
                sp_predict = _fit_spline_interp(X_train, y_train, num_knots=nk, method=meth)
                if sp_predict is not None:
                    pred = sp_predict(X_train)
                    rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_predict = sp_predict

    # Try Gaussian Process with different length scales
    if _remaining() > 15.0:
        for ls in [0.5, 1.0, 2.0, 5.0]:
            if _remaining() < 5.0:
                break
            gp_predict = _fit_gaussian_process(X_train, y_train, length_scale=ls)
            if gp_predict is not None:
                pred = gp_predict(X_train)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = gp_predict

    # Try Kernel Ridge Regression
    if _remaining() > 15.0:
        for ls in [0.5, 1.0, 2.0]:
            if _remaining() < 5.0:
                break
            kr_predict = _fit_kernel_ridge(X_train, y_train, length_scale=ls)
            if kr_predict is not None:
                pred = kr_predict(X_train)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = kr_predict

    # Try RBF with optimization
    if _remaining() > 25.0:
        rbf_predict = _fit_rbf_fixed(X_train, y_train)
        if rbf_predict is not None:
            pred = rbf_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = rbf_predict

    # Try MLP
    if _remaining() > 30.0:
        mlp_predict = _fit_mlp(X_train, y_train)
        if mlp_predict is not None:
            pred = mlp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = mlp_predict

    # Try Neural ODE
    if _remaining() > 40.0:
        for n_steps in [20, 50, 100]:
            if _remaining() < 10.0:
                break
            for hidden_dim in [4, 8, 16]:
                if _remaining() < 5.0:
                    break
                for epochs in [100, 300]:
                    if _remaining() < 3.0:
                        break
                    try:
                        nod_predict = _fit_neural_ode(X_train, y_train, n_steps=n_steps, 
                                                        hidden_dim=hidden_dim, epochs=epochs)
                        if nod_predict is not None:
                            pred = nod_predict(X_train)
                            if np.all(np.isfinite(pred)):
                                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_predict = nod_predict
                    except Exception:
                        pass

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
