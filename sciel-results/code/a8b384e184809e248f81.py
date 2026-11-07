# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import make_interp_spline

_TOTAL_DEADLINE = time.time() + 320.0  # safety margin under 330s


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
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for p in range(1, degree + 1):
                    for q in range(1, degree + 1 - p + 1):
                        if p + q <= degree:
                            cols.append((X[:, i] ** p) * (X[:, j] ** q))
    return np.column_stack(cols)


def _trig_features(X, max_freq=8):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, max_freq + 1):
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
        for s in [0.5, 1.0, 1.5, 2.0]:
            cols.append(np.exp(s * x))
            cols.append(np.exp(-s * x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (x + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-3))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
    return np.column_stack(cols)


def _mixed_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for deg in range(1, 8):
            cols.append(x ** deg)
        for f in range(1, 6):
            cols.append(np.sin(f * x))
            cols.append(np.cos(f * x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-3))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _rational_features(X, num_deg=6, den_deg=5):
    """Features for rational function P(x)/Q(x) as linear system.
    P(x)*Q(x) = y*Q(x) where P and Q are polynomials.
    We set Q(0)=1 (normalize) and solve for coefficients.
    """
    d = X.shape[1]
    if d != 1:
        # Fallback to mixed features for multi-dim
        return _mixed_features(X)
    x = X[:, 0]
    n = len(x)
    
    # P(x) = sum_{i=0}^{num_deg} p_i x^i
    # Q(x) = 1 + sum_{i=1}^{den_deg} q_i x^i
    # P(x) = y * Q(x)
    # sum_{i=0}^{num_deg} p_i x^i = y * (1 + sum_{i=1}^{den_deg} q_i x^i)
    # sum_{i=0}^{num_deg} p_i x^i - sum_{i=1}^{den_deg} y*q_i*x^i = y
    
    # This is nonlinear in (p, q, y). Instead, use the linear formulation:
    # p_0 + p_1 x + ... + p_{num_deg} x^{num_deg} = y * (1 + q_1 x + ... + q_{den_deg} x^{den_deg})
    # Rearranging: p_0 + p_1 x + ... + p_{num_deg} x^{num_deg} - y*q_1*x - ... - y*q_{den_deg}*x^{den_deg} = y
    # This is still nonlinear. 
    # Alternative: treat it as P(x) - y*Q(x) = 0 where Q(0)=1.
    # We can use the fact that for each data point: P(x_i) = y_i * Q(x_i)
    # This gives a system that's linear in p and q if we treat y_i as known.
    # Actually: p_0 + p_1 x_i + ... + p_{num_deg} x_i^{num_deg} = y_i * (1 + q_1 x_i + ... + q_{den_deg} x_i^{den_deg})
    # => p_0 + p_1 x_i + ... + p_{num_deg} x_i^{num_deg} - y_i*q_1*x_i - ... - y_i*q_{den_deg}*x_i^{den_deg} = y_i
    # This is linear in the unknowns [p_0, ..., p_{num_deg}, q_1, ..., q_{den_deg}]!
    
    # Build the design matrix
    num_params = (num_deg + 1) + den_deg  # p_0..p_{num_deg} and q_1..q_{den_deg}
    F = np.zeros((n, num_params))
    
    # P part: columns 0..num_deg
    for i in range(num_deg + 1):
        F[:, i] = x ** i
    
    # Q part: columns (num_deg+1)..(num_deg+den_deg), corresponding to q_1..q_{den_deg}
    for i in range(1, den_deg + 1):
        F[:, num_deg + i] = -y * (x ** i)
    
    # RHS is y
    rhs = y
    
    return F, rhs, num_deg, den_deg


def _fit_rational(X, y, num_deg=6, den_deg=5):
    """Fit rational function P(x)/Q(x) using linear least squares."""
    try:
        d = X.shape[1]
        if d != 1:
            return None
        x = X[:, 0]
        n = len(x)
        
        num_params = (num_deg + 1) + den_deg
        F = np.zeros((n, num_params))
        
        for i in range(num_deg + 1):
            F[:, i] = x ** i
        
        for i in range(1, den_deg + 1):
            F[:, num_deg + i] = -y * (x ** i)
        
        rhs = y
        
        # Add small ridge for stability
        F_reg = F.copy()
        F_reg[:, 0] += 1e-8 * np.sqrt(n)
        
        w, *_ = np.linalg.lstsq(F_reg, rhs, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        
        p = w[:num_deg + 1]
        q = w[num_deg + 1:]  # q_1..q_{den_deg}
        
        # Check that Q(x) doesn't have zeros in the training range
        q_full = np.concatenate([[1.0], q])
        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        x_check = np.linspace(x_min, x_max, 200)
        Q_vals = np.polyval(q_full[::-1], x_check)
        
        if np.any(np.abs(Q_vals) < 1e-6):
            return None  # Pole in range, skip
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            P_vals = np.polyval(p[::-1], xp)
            Q_vals_p = np.polyval(q_full[::-1], xp)
            result = P_vals / Q_vals_p
            return np.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)
        
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_ols(F, y, ridge_alpha=1e-6):
    try:
        n, p = F.shape
        F_reg = F.copy()
        F_reg[:, 0] += ridge_alpha * np.sqrt(n)
        w, *_ = np.linalg.lstsq(F_reg, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _rmse(F, w, y):
    pred = F @ w
    return np.sqrt(np.mean((pred - y) ** 2))


def _fit_spline(X, y, num_knots=15):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < num_knots + 4:
            num_knots = max(4, n - 4)
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
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


def _fit_mlp(X, y, hidden_sizes=(32, 32), lr=0.005, epochs=300, seed=42):
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


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        rmse = _rmse(F, w, y)
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _fit_gaussian_process(X, y, length_scales=None, noise=1e-6):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        if length_scales is None:
            length_scales = np.ones(d)
        
        def kernel(X1, X2):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scales[np.newaxis, np.newaxis, :]**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(Xn, Xn) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except:
            K += 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K)
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn)
            preds = Ks @ alpha
            return preds.flatten()
        return predict
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
    for deg in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
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

    # Try rational function P(x)/Q(x) - KEY ADDITION
    if _remaining() > 15.0 and d == 1:
        for num_deg in [3, 5, 7]:
            for den_deg in [2, 4, 6]:
                if _remaining() < 5.0:
                    break
                rat_predict = _fit_rational(X_train, y_train, num_deg, den_deg)
                if rat_predict is not None:
                    pred = rat_predict(X_train)
                    rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_predict = rat_predict

    # Try spline
    if _remaining() > 10.0 and d == 1:
        sp_predict = _fit_spline(X_train, y_train)
        if sp_predict is not None:
            pred = sp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = sp_predict

    # Try Gaussian Process
    if _remaining() > 15.0:
        gp_predict = _fit_gaussian_process(X_train, y_train)
        if gp_predict is not None:
            pred = gp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = gp_predict

    # Try MLP
    if _remaining() > 30.0:
        mlp_predict = _fit_mlp(X_train, y_train)
        if mlp_predict is not None:
            pred = mlp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = mlp_predict

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
