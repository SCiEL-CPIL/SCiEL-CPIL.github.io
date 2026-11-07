"""Symbolic regression over 4 hidden equations using a rich basis with ridge regression."""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# Global time guard
_START_TIME = None
TIME_LIMIT = 300.0  # wall-clock seconds
SAFETY_MARGIN = 15.0

def _time_left():
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _rich_basis(X, poly_degree=6, n_trig=4, n_exp=3):
    """Create a rich combined basis: polynomial + trig + exp + log + tanh + abs."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    
    if d == 1:
        x = X[:, 0]
        # Polynomial features
        for k in range(1, poly_degree + 1):
            cols.append(x ** k)
        # Trig features
        for k in range(1, n_trig + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        # Exp features (clipped to avoid overflow)
        x_clipped = np.clip(x, -10, 10)
        for k in range(1, n_exp + 1):
            cols.append(np.exp(k * x_clipped / n_exp))
        # Log features (for positive x)
        x_pos = np.maximum(x, 1e-6)
        cols.append(np.log(x_pos))
        cols.append(np.log1p(np.abs(x)))
        # Tanh features
        cols.append(np.tanh(x))
        cols.append(np.tanh(2 * x))
        # Abs features
        cols.append(np.abs(x))
        cols.append(np.abs(x) ** 0.5)
    elif d == 2:
        x1, x2 = X[:, 0], X[:, 1]
        # Polynomial features (bilinear)
        for i in range(1, 4):
            for j in range(0, 4 - i):
                cols.append((x1 ** i) * (x2 ** j))
        # Trig features
        for k in range(1, n_trig + 1):
            cols.append(np.sin(k * x1))
            cols.append(np.cos(k * x1))
            cols.append(np.sin(k * x2))
            cols.append(np.cos(k * x2))
        # Exp features
        x1_clipped = np.clip(x1, -10, 10)
        x2_clipped = np.clip(x2, -10, 10)
        cols.append(np.exp(x1_clipped / n_exp))
        cols.append(np.exp(x2_clipped / n_exp))
        # Tanh features
        cols.append(np.tanh(x1))
        cols.append(np.tanh(x2))
        # Abs features
        cols.append(np.abs(x1))
        cols.append(np.abs(x2))
    else:
        # General case: linear + quadratic
        for i in range(d):
            cols.append(X[:, i])
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)

def _fit_ridge_rich_basis(X, y, alpha=1.0):
    """Fit ridge regression on rich basis. Returns predict callable."""
    Xf = _rich_basis(X)
    # Center and scale y for numerical stability
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    yn = (y - y_mean) / y_std
    
    # Add regularization
    n_features = Xf.shape[1]
    A = Xf.T @ Xf + alpha * np.eye(n_features)
    b = Xf.T @ yn
    
    try:
        w = np.linalg.solve(A, b)
    except:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    
    def predict(X_in):
        Xf_in = _rich_basis(X_in)
        return (Xf_in @ w) * y_std + y_mean
    return predict

def _fit_rbf_kernel(X, y):
    """Fit RBF kernel regression with selected length scales."""
    Xn = X - X.mean(axis=0)
    Sn = Xn.std(axis=0) + 1e-8
    Xn = Xn / Sn
    yn = y - y.mean()
    
    best_gamma = None
    best_c = None
    best_train_mse = np.inf
    
    for gamma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if _time_left() < 5:
            break
        diff = Xn[:, None, :] - Xn[None, :, :]
        sqdist = np.sum(diff ** 2, axis=-1)
        K = np.exp(-gamma * sqdist) + 1e-6 * np.eye(len(Xn))
        try:
            c = np.linalg.solve(K, yn)
            train_pred = (np.exp(-gamma * sqdist)) @ c
            train_mse = np.mean((train_pred - yn) ** 2)
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                best_gamma = gamma
                best_c = c.copy()
        except:
            continue
    
    if best_c is not None:
        def predict(X_in, Xn_tr=Xn, c=best_c, gam=best_gamma, y_mean=y.mean(), X_mean=X.mean(axis=0), X_std=Sn):
            Xn_in = (X_in - X_mean) / X_std
            diff = Xn_in[:, None, :] - Xn_tr[None, :, :]
            sqdist = np.sum(diff ** 2, axis=-1)
            K = np.exp(-gam * sqdist)
            return K @ c + y_mean
        return predict
    else:
        # Fallback to linear
        Xn = X - X.mean(axis=0)
        Sn = Xn.std(axis=0) + 1e-8
        Xn = Xn / Sn
        X_ext = np.column_stack([np.ones(len(Xn)), Xn])
        lam = 1e-6
        try:
            w = np.linalg.solve(X_ext.T @ X_ext + lam * np.eye(X_ext.shape[1]), X_ext.T @ y)
        except:
            w = np.zeros(X_ext.shape[1])
        def predict(X_in):
            Xn_in = (X_in - X.mean(axis=0)) / Sn
            X_ext_in = np.column_stack([np.ones(len(Xn_in)), Xn_in])
            return X_ext_in @ w
        return predict

def _fit_poly(X, y, degree):
    """Fit polynomial regression of given degree. Returns predict callable."""
    d = X.shape[1]
    if d == 1:
        x = X[:, 0]
        cols = [np.ones(len(x))]
        for k in range(1, degree + 1):
            cols.append(x ** k)
        Xf = np.column_stack(cols)
    elif d == 2:
        x1, x2 = X[:, 0], X[:, 1]
        cols = [np.ones(len(x1))]
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                cols.append((x1 ** i) * (x2 ** j))
        Xf = np.column_stack(cols)
    else:
        cols = [np.ones(len(X))]
        for i in range(d):
            cols.append(X[:, i])
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        Xf = np.column_stack(cols)
    
    try:
        w, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    except:
        w = np.zeros(Xf.shape[1])
    
    def predict(X_in):
        d_in = X_in.shape[1]
        if d_in == 1:
            x = X_in[:, 0]
            cols = [np.ones(len(x))]
            for k in range(1, degree + 1):
                cols.append(x ** k)
            Xf_in = np.column_stack(cols)
        elif d_in == 2:
            x1, x2 = X_in[:, 0], X_in[:, 1]
            cols = [np.ones(len(x1))]
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    cols.append((x1 ** i) * (x2 ** j))
            Xf_in = np.column_stack(cols)
        else:
            cols = [np.ones(len(X_in))]
            for i in range(d_in):
                cols.append(X_in[:, i])
            for i in range(d_in):
                for j in range(i, d_in):
                    cols.append(X_in[:, i] * X_in[:, j])
            Xf_in = np.column_stack(cols)
        return Xf_in @ w
    return predict

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Ensure finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    candidates = []
    
    # Rich basis with ridge regression (primary candidate)
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        if _time_left() < 10:
            break
        try:
            pred = _fit_ridge_rich_basis(X, y, alpha=alpha)
            train_mse = np.mean((pred(X) - y) ** 2) / (np.var(y) + 1e-12)
            candidates.append((train_mse, pred, f"rich_ridge_a{alpha}"))
        except:
            pass
    
    # Try polynomial fits of various degrees
    for deg in [1, 2, 3, 4, 5, 6]:
        if _time_left() < 10:
            break
        try:
            pred = _fit_poly(X, y, deg)
            train_mse = np.mean((pred(X) - y) ** 2) / (np.var(y) + 1e-12)
            candidates.append((train_mse, pred, f"poly{deg}"))
        except:
            pass
    
    # Try RBF kernel
    if _time_left() > 15:
        try:
            pred = _fit_rbf_kernel(X, y)
            train_mse = np.mean((pred(X) - y) ** 2) / (np.var(y) + 1e-12)
            candidates.append((train_mse, pred, "rbf"))
        except:
            pass
    
    # Pick best by training MSE (proxy for test performance)
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_mse, best_pred, best_name = candidates[0]
        return best_pred
    else:
        # Fallback: constant predictor
        y_mean = y.mean()
        return lambda X_in: np.full(len(X_in), y_mean)
