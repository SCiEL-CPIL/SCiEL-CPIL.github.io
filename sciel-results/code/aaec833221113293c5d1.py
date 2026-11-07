"""Symbolic regression over 10 hidden equations: Expanded feature search + RBF fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.linalg

_START = time.time()
_DEADLINE = 300.0  # safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _make_predict_rbf(X_train, y_train, gamma, alpha):
    """Create a predict function for RBF Kernel Ridge Regression."""
    # Precompute kernel matrix for training
    # K[i, j] = exp(-gamma * ||X_i - X_j||^2)
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    K = np.exp(-gamma * sq_dists)
    
    # Regularization
    K_reg = K + alpha * np.eye(len(X_train))
    
    try:
        # Solve for weights
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        # Fallback to pseudo-inverse if singular
        w = np.linalg.pinv(K_reg) @ y_train
        
    def predict(X):
        diff_test = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test**2, axis=-1)
        K_test = np.exp(-gamma * sq_dists_test)
        return K_test @ w
        
    return predict

def _make_predict_linear(basis_func, w):
    """Create a predict function for linear combination of basis features."""
    def predict(X):
        A = basis_func(X)
        return A @ w
    return predict

def _try_exact_fit(basis_func, X, y, tol=1e-6):
    """
    Try to fit y using linear combination of basis_func(X).
    Returns (w, err) if fit is good, else None.
    """
    try:
        A = basis_func(X)
        # Check for finite values
        if not np.all(np.isfinite(A)):
            return None
        
        # Use lstsq
        w, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        # Calculate prediction error
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
            
        mse = np.mean((pred - y)**2)
        # Scale tolerance based on y magnitude
        scale = max(1.0, np.max(np.abs(y)))
        
        if mse <= (tol * scale)**2:
            return w, mse
    except Exception:
        pass
    return None

# --- Basis Function Generators ---

def _basis_poly2(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)

def _basis_poly3(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    for i in range(d):
        cols.append(X[:, i] ** 3)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_poly4(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
        cols.append(X[:, j] ** 4)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
            cols.append(X[:, i] ** 2 * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_rich(X):
    """Includes polynomials and common transcendental functions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    
    # Polynomials up to degree 2
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            
    # Transcendentals on individual variables
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        # Exp can blow up, but inputs are in [1, 10] max, so exp(10) is ~22000, manageable.
        # However, if the function is not exp, this feature will just have a coefficient of 0.
        # To avoid numerical issues with lstsq if values are huge, we might normalize or skip.
        # Let's include exp, log, sinh, cosh.
        cols.append(np.exp(x))
        # Log is defined for x>0. Inputs are >=1, so safe.
        cols.append(np.log(x))
        cols.append(np.sinh(x))
        cols.append(np.cosh(x))
        
    # Some simple interactions with transcendentals? Might be too many.
    # Let's stick to univariate transcendentals + poly2 for now to keep matrix size reasonable.
    
    return np.column_stack(cols)

def _basis_trig_poly(X):
    """Polynomials and trigonometric functions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j])) # Example interaction
    return np.column_stack(cols)

_CANDIDATE_BASES = [
    _basis_poly2,
    _basis_poly3,
    _basis_poly4,
    _basis_rich,
    _basis_trig_poly
]

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    n_samples = len(X)
    d = X.shape[1]
    
    # 1. Try Exact Fit with various bases
    best_exact_predict = None
    best_exact_err = float('inf')
    
    for basis_func in _CANDIDATE_BASES:
        if _remaining() < 10.0:
            break
        try:
            result = _try_exact_fit(basis_func, X, y)
            if result is not None:
                w, err = result
                if err < best_exact_err:
                    best_exact_err = err
                    best_exact_predict = _make_predict_linear(basis_func, w)
        except Exception:
            continue
            
    if best_exact_predict is not None:
        return best_exact_predict
        
    # 2. Fallback: RBF Kernel Ridge Regression
    # Grid search for hyperparameters
    # Inputs are in ranges like [1, 3], [1, 5], [1, 10].
    # Distances can be up to ~10-20.
    # gamma = 1 / (2 * sigma^2).
    # If sigma is around 1, gamma ~ 0.5. If sigma is 0.5, gamma ~ 2.
    
    best_rbf_predict = None
    best_rbf_err = float('inf')
    
    # We use a small holdout set for validation if time permits, 
    # but with 200 points, we can just minimize training error 
    # with strong regularization, or use CV.
    # Given the time limit, let's do a quick grid search on training error 
    # but penalize complexity via alpha.
    
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    alphas = [1e-6, 1e-4, 1e-2, 0.1, 1.0]
    
    # To save time, we can subsample the grid or just try a few promising ones.
    # Let's try all combinations. 7 * 5 = 35 fits.
    # Each fit is O(N^3) or O(N^2 * N) solve. N=200. 200^3 = 8e6 ops. Very fast.
    
    for gamma in gammas:
        if _remaining() < 5.0:
            break
        for alpha in alphas:
            try:
                predict_func = _make_predict_rbf(X, y, gamma, alpha)
                pred = predict_func(X)
                err = np.mean((pred - y)**2)
                if err < best_rbf_err:
                    best_rbf_err = err
                    best_rbf_predict = predict_func
            except Exception:
                continue
                
    if best_rbf_predict is not None:
        return best_rbf_predict
        
    # 3. Ultimate Fallback: Simple Linear Regression
    try:
        A = _basis_poly2(X)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return _make_predict_linear(_basis_poly2, w)
    except Exception:
        # Constant prediction
        mean_y = np.mean(y)
        return lambda X: np.full(len(X), mean_y)

# EVOLVE-BLOCK-END
