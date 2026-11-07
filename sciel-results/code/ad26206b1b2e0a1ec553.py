"""Symbolic regression with rich basis, adaptive degree, and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_features_1d(x, max_poly=12, max_trig=8):
    """Build a rich feature matrix for 1D input."""
    n = len(x)
    cols = [np.ones(n)]
    
    # Polynomial terms
    for k in range(1, max_poly + 1):
        cols.append(x ** k)
    
    # Trigonometric terms
    for k in range(1, max_trig + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    
    # Exponential and log-like terms (guarded for domain issues)
    # exp(x) can blow up, so use scaled versions
    cols.append(np.exp(0.5 * x))
    cols.append(np.exp(-0.5 * x))
    
    # Sqrt for positive domains
    pos_mask = x > 0
    if np.any(pos_mask):
        sqrt_x = np.zeros(n)
        sqrt_x[pos_mask] = np.sqrt(x[pos_mask])
        cols.append(sqrt_x)
        cols.append(x * sqrt_x)
    
    # Abs and sign
    cols.append(np.abs(x))
    cols.append(np.sign(x) * x)  # same as x, but keep for clarity
    
    # Rational-like: x/(1+x^2), etc.
    denom = 1 + x ** 2
    cols.append(x / denom)
    cols.append(x ** 2 / denom)
    
    # Additional trig combos
    cols.append(np.sin(x) * np.cos(x))
    cols.append(np.sin(x) ** 2)
    cols.append(np.cos(x) ** 2)
    
    return np.column_stack(cols)


def _build_features_2d(X, max_poly=4):
    """Build features for multi-dimensional input (up to quadratic/cubic)."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n)]
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
    
    # Quadratic terms
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    
    # Cubic terms
    for i in range(d):
        for j in range(d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Trig for each dimension
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    
    return np.column_stack(cols)


def _ridge_solve(A, y, alpha):
    """Ridge regression solver."""
    n, p = A.shape
    # A^T A + alpha * I
    AtA = A.T @ A
    AtA += alpha * np.eye(p)
    Aty = A.T @ y
    try:
        w = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
    return w


def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    d = X_train.shape[1]
    
    # For 1D, use rich basis; for multi-D, use polynomial + trig
    if d == 1:
        x = X_train[:, 0]
        
        # Try increasing complexity with ridge regularization
        best_w = None
        best_score = np.inf
        best_A = None
        best_X = None
        
        # Start with moderate complexity
        for max_poly in [6, 8, 10, 12]:
            if _time_left() < 5:
                break
            A = _build_features_1d(x, max_poly=max_poly, max_trig=8)
            
            # Try several regularization strengths
            for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                if _time_left() < 5:
                    break
                try:
                    w = _ridge_solve(A, y_train, alpha)
                    pred = A @ w
                    score = np.mean((pred - y_train) ** 2)
                    if score < best_score:
                        best_score = score
                        best_w = w
                        best_A = A
                        best_X = x
                except Exception:
                    continue
        
        if best_w is None:
            # Fallback: simple polynomial
            A = np.column_stack([np.ones(len(x)), x, x**2, x**3])
            w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
            best_w = w
            best_A = A
            best_X = x
        
        # Predict function
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            x_pred = X[:, 0] if X.ndim > 1 else X
            # Rebuild features with same max_poly
            # We need to know which max_poly was used; store it
            # For simplicity, rebuild with the same structure
            # Actually, we should store the feature builder params
            # Let's use a closure approach
            pass
        
        # Better: create predict that rebuilds features
        # We need to track which max_poly was used
        # Let's refactor: store the feature builder
        def make_predict_1d(x_train, max_poly, max_trig, w):
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                x_pred = X[:, 0] if X.ndim > 1 else X
                A = _build_features_1d(x_pred, max_poly=max_poly, max_trig=max_trig)
                return A @ w
            return predict
        
        # Determine max_poly used
        # We'll just use the best one found
        # Re-derive: find which max_poly gave best
        # Simpler: just use max_poly=10 as default if best found
        # Actually, let's track it properly
        
        # Redo with tracking
        best_w = None
        best_score = np.inf
        best_max_poly = 10
        best_max_trig = 8
        
        for max_poly in [6, 8, 10, 12]:
            if _time_left() < 5:
                break
            A = _build_features_1d(x, max_poly=max_poly, max_trig=8)
            for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                if _time_left() < 5:
                    break
                try:
                    w = _ridge_solve(A, y_train, alpha)
                    pred = A @ w
                    score = np.mean((pred - y_train) ** 2)
                    if score < best_score:
                        best_score = score
                        best_w = w
                        best_max_poly = max_poly
                        best_max_trig = 8
                except Exception:
                    continue
        
        if best_w is None:
            A = np.column_stack([np.ones(len(x)), x, x**2, x**3])
            best_w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
            best_max_poly = 3
            best_max_trig = 0
        
        return make_predict_1d(x, best_max_poly, best_max_trig, best_w)
    
    else:
        # Multi-dimensional
        A = _build_features_2d(X_train, max_poly=3)
        w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            A_pred = _build_features_2d(X, max_poly=3)
            return A_pred @ w
        
        return predict
# EVOLVE-BLOCK-END
