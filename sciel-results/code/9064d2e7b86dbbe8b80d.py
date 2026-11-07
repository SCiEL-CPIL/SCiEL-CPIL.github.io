# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_rich_basis(x):
    """Build a rich set of basis functions for 1D input."""
    n = len(x)
    cols = [np.ones(n)]
    
    # Polynomial terms up to degree 8 (higher degrees can cause numerical issues without strong reg)
    for k in range(1, 9):
        cols.append(x ** k)
    
    # Trigonometric terms
    for freq in [0.5, 1, 2, 3]:
        cols.append(np.sin(freq * x))
        cols.append(np.cos(freq * x))
    
    # Exponential terms (clipped for stability)
    cols.append(np.exp(np.clip(x, -5, 5)))
    cols.append(np.exp(np.clip(-x, -5, 5)))
    
    # Log terms (handle x<=0)
    log_x = np.where(x > 1e-8, np.log(np.maximum(x, 1e-8)), 0.0)
    cols.append(log_x)
    cols.append(log_x ** 2)
    
    # Absolute value and sign
    cols.append(np.abs(x))
    cols.append(np.sign(x))
    
    # Hyperbolic functions
    cols.append(np.tanh(x))
    cols.append(np.arctan(x))
    
    # Rational-like functions
    cols.append(x / (1 + np.abs(x)))
    cols.append(x / (1 + x ** 2))
    
    return np.column_stack(cols)


def _rbf_kernel(X, Y, gamma=1.0):
    """Compute RBF kernel between X and Y."""
    X2 = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y2 = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    dist = X2 + Y2 - 2 * X @ Y.T
    dist = np.maximum(dist, 0)
    return np.exp(-gamma * dist)


def _fit_ridge(X, y, lam):
    """Fit ridge regression."""
    n_features = X.shape[1]
    try:
        A = X.T @ X
        b = X.T @ y
        reg = A + lam * np.eye(n_features)
        w = np.linalg.solve(reg, b)
        return w
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return w


def _fit_rbf(X, y, gamma, lam):
    """Fit RBF kernel ridge regression."""
    K = _rbf_kernel(X, X, gamma)
    n = len(X)
    try:
        w = np.linalg.solve(K + lam * np.eye(n), y)
        return w
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(K + lam * np.eye(n), y, rcond=None)
        return w


def _kfold_cv_ridge(X, y, lam, k=5):
    """K-Fold Cross Validation for Ridge Regression."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    errors = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        w = _fit_ridge(X_train, y_train, lam)
        pred = X_val @ w
        errors.append(np.mean((pred - y_val) ** 2))
    return np.mean(errors)


def _kfold_cv_rbf(X, y, gamma, lam, k=5):
    """K-Fold Cross Validation for RBF Kernel Ridge Regression."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    errors = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        w = _fit_rbf(X_train, y_train, gamma, lam)
        K_val = _rbf_kernel(X_val, X_train, gamma)
        pred = K_val @ w
        errors.append(np.mean((pred - y_val) ** 2))
    return np.mean(errors)


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n, d = X_train.shape
    
    # Normalize input to [-1, 1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1.0
    X_norm = (X_train - x_min) / x_range * 2 - 1
    
    best_cv_score = np.inf
    best_predict_fn = None
    
    # Strategy 1: Ridge Regression on Rich Basis
    if _time_left() > 10:
        X_basis = _build_rich_basis(X_norm[:, 0])
        lams = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        for lam in lams:
            if _time_left() < 5:
                break
            try:
                cv_score = _kfold_cv_ridge(X_basis, y_train, lam)
                if cv_score < best_cv_score:
                    best_cv_score = cv_score
                    w_best = _fit_ridge(X_basis, y_train, lam)
                    
                    def predict_basis_fn(X, w=w_best, x_min=x_min, x_range=x_range):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        Xn = (X - x_min) / x_range * 2 - 1
                        Xf = _build_rich_basis(Xn[:, 0])
                        return Xf @ w
                    
                    best_predict_fn = predict_basis_fn
            except Exception:
                continue
    
    # Strategy 2: RBF Kernel Ridge Regression
    if _time_left() > 10:
        gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        lams = [1e-4, 1e-2, 1e-1, 1.0, 10.0]
        for gamma in gammas:
            if _time_left() < 5:
                break
            for lam in lams:
                if _time_left() < 2:
                    break
                try:
                    cv_score = _kfold_cv_rbf(X_norm, y_train, gamma, lam)
                    if cv_score < best_cv_score:
                        best_cv_score = cv_score
                        w_best = _fit_rbf(X_norm, y_train, gamma, lam)
                        
                        def predict_rbf_fn(X, w=w_best, gamma=gamma, X_train=X_norm, x_min=x_min, x_range=x_range):
                            X = np.asarray(X, dtype=np.float64)
                            if X.ndim == 1:
                                X = X.reshape(-1, 1)
                            Xn = (X - x_min) / x_range * 2 - 1
                            K_val = _rbf_kernel(Xn, X_train, gamma)
                            return K_val @ w
                        
                        best_predict_fn = predict_rbf_fn
                except Exception:
                    continue
    
    # Strategy 3: Polynomial Ridge (if rich basis failed or for redundancy)
    if _time_left() > 5:
        for deg in [3, 5, 7]:
            if _time_left() < 2:
                break
            try:
                # Vandermonde matrix for polynomial features
                X_poly = np.vander(X_norm[:, 0], N=deg+1, increasing=True)
                lams = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
                for lam in lams:
                    if _time_left() < 1:
                        break
                    cv_score = _kfold_cv_ridge(X_poly, y_train, lam)
                    if cv_score < best_cv_score:
                        best_cv_score = cv_score
                        w_best = _fit_ridge(X_poly, y_train, lam)
                        
                        def predict_poly_fn(X, w=w_best, deg=deg, x_min=x_min, x_range=x_range):
                            X = np.asarray(X, dtype=np.float64)
                            if X.ndim == 1:
                                X = X.reshape(-1, 1)
                            Xn = (X - x_min) / x_range * 2 - 1
                            Xf = np.vander(Xn[:, 0], N=deg+1, increasing=True)
                            return Xf @ w
                        
                        best_predict_fn = predict_poly_fn
            except Exception:
                continue

    # Fallback: Simple Linear Regression
    if best_predict_fn is None:
        def predict_linear(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            Xn = (X - x_min) / x_range * 2 - 1
            x = Xn[:, 0]
            a = np.dot(x - x.mean(), y_train - y_train.mean()) / (np.dot(x - x.mean(), x - x.mean()) + 1e-8)
            b = y_train.mean() - a * x.mean()
            return a * x + b
        best_predict_fn = predict_linear
    
    return best_predict_fn
# EVOLVE-BLOCK-END
