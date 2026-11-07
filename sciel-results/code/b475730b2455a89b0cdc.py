# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import solve

_TOTAL_DEADLINE = time.time() + 310.0


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
            if y.ndim == 0:
                y = np.array([y])
            else:
                y = y.ravel()
            if y.shape[0] != X.shape[0]:
                y = np.resize(y, X.shape[0])
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _rmse(pred, y):
    """Compute RMSE, returning inf if non-finite."""
    try:
        pred = np.asarray(pred, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(y)):
            return np.inf
        return np.sqrt(np.mean((pred - y) ** 2))
    except Exception:
        return np.inf


def _fit_cubic_spline(X, y):
    """Fit a natural cubic spline to the data."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 4:
            return None
        # Remove duplicate x values if any
        unique_x, unique_idx = np.unique(xs, return_index=True)
        if len(unique_x) < 4:
            return None
        ys_unique = ys[unique_idx]
        s = CubicSpline(unique_x, ys_unique)
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


def _fit_pchip(X, y):
    """Fit a PCHIP interpolator to the data."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 3:
            return None
        unique_x, unique_idx = np.unique(xs, return_index=True)
        if len(unique_x) < 3:
            return None
        ys_unique = ys[unique_idx]
        s = PchipInterpolator(unique_x, ys_unique)
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


def _fit_kernel_ridge(X, y, length_scales=None, alphas=None):
    """Kernel Ridge Regression with RBF kernel. Returns list of (predict_fn, rmse)."""
    results = []
    n, d = X.shape
    if length_scales is None:
        length_scales = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    if alphas is None:
        alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma
    
    for ls in length_scales:
        if _remaining() < 5.0:
            break
        for alpha in alphas:
            if _remaining() < 2.0:
                break
            try:
                diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2) / (ls ** 2)
                K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
                
                try:
                    w = solve(K, y)
                except Exception:
                    try:
                        w = np.linalg.lstsq(K, y, rcond=None)[0]
                    except Exception:
                        continue
                
                if not np.all(np.isfinite(w)):
                    continue
                
                pred = (np.exp(-0.5 * sq_dist) @ w)
                rmse = _rmse(pred, y)
                
                def predict(Xp, mu=mu, sigma=sigma, Xn=Xn, w=w, ls=ls):
                    Xp = np.asarray(Xp, dtype=np.float64)
                    if Xp.ndim == 1:
                        Xp = Xp.reshape(1, -1)
                    Xpn = (Xp - mu) / sigma
                    diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
                    sq_dist = np.sum(diff ** 2, axis=2) / (ls ** 2)
                    Ks = np.exp(-0.5 * sq_dist)
                    preds = Ks @ w
                    return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
                
                results.append((_safe_predict(predict), rmse))
            except Exception:
                continue
    return results


def _fit_poly(X, y, max_degree=20):
    """Fit polynomial regression with increasing degrees."""
    results = []
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-9:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2.0 * (x - x_min) / x_range - 1.0
    
    for deg in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]:
        if _remaining() < 3.0:
            break
        try:
            F = np.vander(x_norm, N=deg + 1, increasing=True)
            # Ridge regularization for stability
            alpha = 1e-6
            A = F.T @ F + alpha * np.eye(deg + 1)
            b = F.T @ y
            w = solve(A, b)
            if not np.all(np.isfinite(w)):
                continue
            pred = F @ w
            rmse = _rmse(pred, y)
            
            def predict(Xp, x_min=x_min, x_range=x_range, w=w, deg=deg):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                if x_range < 1e-9:
                    xp_norm = np.zeros_like(xp)
                else:
                    xp_norm = 2.0 * (xp - x_min) / x_range - 1.0
                Fp = np.vander(xp_norm, N=deg + 1, increasing=True)
                return np.nan_to_num(Fp @ w, nan=0.0, posinf=1e12, neginf=-1e12)
            
            results.append((_safe_predict(predict), rmse))
        except Exception:
            continue
    return results


def _fit_fourier(X, y, max_harmonics=20):
    """Fit Fourier series regression."""
    results = []
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-9:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2.0 * np.pi * (x - x_min) / x_range
    
    for nh in [2, 4, 6, 8, 10, 12, 15, 20]:
        if _remaining() < 3.0:
            break
        try:
            cols = [np.ones(x.shape[0])]
            for k in range(1, nh + 1):
                cols.append(np.sin(k * x_norm))
                cols.append(np.cos(k * x_norm))
            F = np.column_stack(cols)
            alpha = 1e-6
            A = F.T @ F + alpha * np.eye(F.shape[1])
            b = F.T @ y
            w = solve(A, b)
            if not np.all(np.isfinite(w)):
                continue
            pred = F @ w
            rmse = _rmse(pred, y)
            
            def predict(Xp, x_min=x_min, x_range=x_range, w=w, nh=nh):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                if x_range < 1e-9:
                    xp_norm = np.zeros_like(xp)
                else:
                    xp_norm = 2.0 * np.pi * (xp - x_min) / x_range
                cols = [np.ones(xp.shape[0])]
                for k in range(1, nh + 1):
                    cols.append(np.sin(k * xp_norm))
                    cols.append(np.cos(k * xp_norm))
                Fp = np.column_stack(cols)
                return np.nan_to_num(Fp @ w, nan=0.0, posinf=1e12, neginf=-1e12)
            
            results.append((_safe_predict(predict), rmse))
        except Exception:
            continue
    return results


def _fit_mlp(X, y, hidden_sizes=None, epochs=2000, lr=1e-3, batch_size=32):
    """Simple MLP regression using manual gradient descent."""
    results = []
    if hidden_sizes is None:
        hidden_sizes = [[16], [32], [16, 16], [32, 32], [64], [16, 32], [32, 16]]
    
    n, d = X.shape
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std
    
    for hs in hidden_sizes:
        if _remaining() < 10.0:
            break
        try:
            # Initialize weights
            np.random.seed(42)
            layers = [d] + list(hs) + [1]
            W = []
            b = []
            for i in range(len(layers) - 1):
                fan_in, fan_out = layers[i], layers[i + 1]
                W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
                b.append(np.zeros(fan_out))
            
            # Training
            for epoch in range(epochs):
                if _remaining() < 5.0:
                    break
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    batch_idx = idx[start:start + batch_size]
                    Xb = Xn[batch_idx]
                    yb = yn[batch_idx]
                    
                    # Forward
                    cache = [Xb]
                    a = Xb
                    for i in range(len(W)):
                        z = a @ W[i] + b[i]
                        if i < len(W) - 1:
                            a = np.tanh(z)
                        else:
                            a = z
                        cache.append(a)
                    
                    # Backward
                    delta = a[-1] - yb.reshape(-1, 1)
                    for i in range(len(W) - 1, -1, -1):
                        dW = cache[i].T @ delta
                        db = delta.sum(axis=0)
                        if i > 0:
                            delta = (delta @ W[i].T) * (1 - cache[i] ** 2)
                        W[i] -= lr * dW
                        b[i] -= lr * db
            
            # Evaluate
            a = Xn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
            pred = a.ravel() * y_std + y_mean
            rmse = _rmse(pred, y)
            
            def predict(Xp, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, W=W, b=b, hs=hs):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                Xpn = (Xp - x_mean) / x_std
                a = Xpn
                for i in range(len(W)):
                    z = a @ W[i] + b[i]
                    if i < len(W) - 1:
                        a = np.tanh(z)
                    else:
                        a = z
                return np.nan_to_num(a.ravel() * y_std + y_mean, nan=0.0, posinf=1e12, neginf=-1e12)
            
            results.append((_safe_predict(predict), rmse))
        except Exception:
            continue
    return results


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = _rmse(np.full(n, const), y_train)
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    all_candidates = []

    # 1. Cubic Spline (often best for noiseless 1D)
    if d == 1 and _remaining() > 5.0:
        sp = _fit_cubic_spline(X_train, y_train)
        if sp is not None:
            pred = sp(X_train)
            rmse = _rmse(pred, y_train)
            all_candidates.append((sp, rmse))

    # 2. PCHIP
    if d == 1 and _remaining() > 5.0:
        sp = _fit_pchip(X_train, y_train)
        if sp is not None:
            pred = sp(X_train)
            rmse = _rmse(pred, y_train)
            all_candidates.append((sp, rmse))

    # 3. Polynomial Regression
    if _remaining() > 10.0:
        poly_results = _fit_poly(X_train, y_train)
        all_candidates.extend(poly_results)

    # 4. Fourier Series
    if _remaining() > 10.0:
        fourier_results = _fit_fourier(X_train, y_train)
        all_candidates.extend(fourier_results)

    # 5. Kernel Ridge Regression
    if _remaining() > 30.0:
        kr_results = _fit_kernel_ridge(X_train, y_train)
        all_candidates.extend(kr_results)

    # 6. MLP
    if _remaining() > 60.0:
        mlp_results = _fit_mlp(X_train, y_train)
        all_candidates.extend(mlp_results)

    # Pick the best
    for predict_fn, rmse in all_candidates:
        if rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict_fn

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
