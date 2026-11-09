"""Symbolic regression via feature-space least squares with rich basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Check if y is constant
    y_var = np.var(y)
    if y_var < 1e-12:
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
    
    # Build a rich feature basis
    # For 1D input: include polynomials up to degree 6, plus trig functions
    # For multi-D: include interactions up to degree 3
    
    def build_features(X):
        feats = []
        n_rows = X.shape[0]
        d = X.shape[1]
        
        # Constant
        feats.append(np.ones(n_rows))
        
        # Linear terms
        for j in range(d):
            feats.append(X[:, j].copy())
        
        # Quadratic and cubic terms
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    feats.append(X[:, i] ** 2)
                    feats.append(X[:, i] ** 3)
                else:
                    feats.append(X[:, i] * X[:, j])
        
        # Higher degree univariate terms (for 1D case)
        if d == 1:
            x = X[:, 0]
            for deg in range(4, 9):
                feats.append(x ** deg)
            # Trigonometric features
            feats.append(np.sin(x))
            feats.append(np.cos(x))
            feats.append(np.sin(2 * x))
            feats.append(np.cos(2 * x))
            feats.append(np.sin(3 * x))
            feats.append(np.cos(3 * x))
            # Exponential-like (tanh)
            feats.append(np.tanh(x))
            feats.append(np.tanh(2 * x))
            # Absolute value
            feats.append(np.abs(x))
            # Reciprocal (careful with zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv = np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0)
            feats.append(inv)
            # Log (for positive x)
            with np.errstate(divide='ignore', invalid='ignore'):
                logx = np.where(x > 1e-8, np.log(x), 0.0)
            feats.append(logx)
        
        return np.column_stack(feats)
    
    F_train = build_features(X)
    
    # Use ridge regression for stability
    # Try multiple lambda values and pick best via cross-validation-like approach
    # But we don't have test data, so use simple heuristic: small ridge
    
    best_w = None
    best_score = np.inf
    
    # Try a few ridge parameters
    for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        if time.time() > deadline:
            break
        try:
            if lam == 0.0:
                w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
            else:
                # Ridge: (F^T F + lam I) w = F^T y
                FtF = F_train.T @ F_train + lam * np.eye(F_train.shape[1])
                Fty = F_train.T @ y
                w = np.linalg.solve(FtF, Fty)
            
            # Compute training MSE to gauge fit
            pred = F_train @ w
            mse = np.mean((pred - y) ** 2)
            nmse = mse / y_var
            
            if np.isfinite(nmse) and nmse < best_score:
                best_score = nmse
                best_w = w.copy()
        except Exception:
            continue
    
    if best_w is None:
        # Fallback: simple linear fit
        F_simple = np.column_stack([np.ones(n), X])
        best_w, *_ = np.linalg.lstsq(F_simple, y, rcond=None)
    
    # Store feature builder for prediction
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        F_test = build_features(X_test)
        return F_test @ best_w
    
    return predict
# EVOLVE-BLOCK-END
