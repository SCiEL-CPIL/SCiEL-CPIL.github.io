"""Improved symbolic regression with adaptive feature basis and model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_basis_1d(x, max_poly=10, max_trig=4, include_exp=True):
    """Build a rich basis for 1D input."""
    n = len(x)
    cols = [np.ones(n)]
    
    # Polynomial terms
    for k in range(1, max_poly + 1):
        cols.append(x ** k)
    
    # Trigonometric terms
    for k in range(1, max_trig + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    
    # Exponential terms (careful with large values)
    if include_exp:
        # Only add if values are bounded
        if np.all(np.abs(x) < 10):
            cols.append(np.exp(x))
            cols.append(np.exp(-x))
    
    return np.column_stack(cols)


def _build_basis_2d(X, max_poly=6):
    """Build basis for 2D input."""
    n = X.shape[0]
    x1, x2 = X[:, 0], X[:, 1]
    cols = [np.ones(n)]
    
    for i in range(1, max_poly + 1):
        for j in range(0, max_poly + 1 - i):
            cols.append(x1 ** i * x2 ** j)
    
    return np.column_stack(cols)


def _build_basis_3d(X, max_poly=5):
    """Build basis for 3D input."""
    n = X.shape[0]
    cols = [np.ones(n)]
    
    for deg in range(1, max_poly + 1):
        for combo in itertools.combinations_with_replacement(range(3), deg):
            term = np.ones(n)
            for idx in combo:
                term = term * X[:, idx]
            cols.append(term)
    
    return np.column_stack(cols)


def _fit_ridge(X, y, alpha=1e-6):
    """Fit ridge regression."""
    n_samples, n_features = X.shape
    # Add regularization
    X_reg = np.hstack([X, np.ones((n_samples, 1))])  # Already have ones in basis
    # Actually, our basis already includes ones, so just use X
    A = X.T @ X
    b = X.T @ y
    A = A + alpha * np.eye(n_features)
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w


def _predict_with_weights(X, w):
    return X @ w


def _compute_nrmse(X, y, w):
    pred = X @ w
    var_y = np.var(y)
    if var_y < 1e-12:
        return 1.0
    mse = np.mean((pred - y) ** 2)
    return mse / var_y


def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    d = X.shape[1]
    n = X.shape[0]
    
    # Build appropriate basis based on dimension
    if d == 1:
        # Try different max_poly values
        best_w = None
        best_nrmse = np.inf
        
        # First, a quick fit with moderate basis
        basis = _build_basis_1d(X, max_poly=8, max_trig=3, include_exp=True)
        
        # Use ridge regression with small alpha
        w = _fit_ridge(basis, y, alpha=1e-8)
        best_w = w
        best_basis = basis
        best_nrmse = _compute_nrmse(basis, y, w)
        
        # Try higher polynomial if time allows
        if _time_left() > 50:
            basis_high = _build_basis_1d(X, max_poly=12, max_trig=4, include_exp=True)
            w_high = _fit_ridge(basis_high, y, alpha=1e-8)
            nrmse_high = _compute_nrmse(basis_high, y, w_high)
            if nrmse_high < best_nrmse:
                best_w = w_high
                best_basis = basis_high
                best_nrmse = nrmse_high
        
        # Try with more trig terms
        if _time_left() > 30:
            basis_trig = _build_basis_1d(X, max_poly=8, max_trig=6, include_exp=True)
            w_trig = _fit_ridge(basis_trig, y, alpha=1e-8)
            nrmse_trig = _compute_nrmse(basis_trig, y, w_trig)
            if nrmse_trig < best_nrmse:
                best_w = w_trig
                best_basis = basis_trig
                best_nrmse = nrmse_trig
        
        return lambda X_test: _predict_with_weights(_build_basis_1d(X_test, max_poly=12, max_trig=6, include_exp=True), best_w)
    
    elif d == 2:
        basis = _build_basis_2d(X, max_poly=6)
        w = _fit_ridge(basis, y, alpha=1e-6)
        
        if _time_left() > 30:
            basis_high = _build_basis_2d(X, max_poly=8)
            w_high = _fit_ridge(basis_high, y, alpha=1e-6)
            if _compute_nrmse(basis_high, y, w_high) < _compute_nrmse(basis, y, w):
                basis = basis_high
                w = w_high
        
        return lambda X_test: _predict_with_weights(_build_basis_2d(X_test, max_poly=8), w)
    
    elif d == 3:
        basis = _build_basis_3d(X, max_poly=4)
        w = _fit_ridge(basis, y, alpha=1e-6)
        
        if _time_left() > 30:
            basis_high = _build_basis_3d(X, max_poly=5)
            w_high = _fit_ridge(basis_high, y, alpha=1e-6)
            if _compute_nrmse(basis_high, y, w_high) < _compute_nrmse(basis, y, w):
                basis = basis_high
                w = w_high
        
        return lambda X_test: _predict_with_weights(_build_basis_3d(X_test, max_poly=5), w)
    
    else:
        # Fallback: linear + pairwise interactions
        cols = [np.ones(n)]
        for j in range(d):
            cols.append(X[:, j])
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
        basis = np.column_stack(cols)
        w = _fit_ridge(basis, y, alpha=1e-6)
        
        def _build_basis_generic(X_test):
            n_test = X_test.shape[0]
            cols = [np.ones(n_test)]
            for j in range(d):
                cols.append(X_test[:, j])
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(X_test[:, i] * X_test[:, j])
            return np.column_stack(cols)
        
        return lambda X_test: _predict_with_weights(_build_basis_generic(X_test), w)
# EVOLVE-BLOCK-END
