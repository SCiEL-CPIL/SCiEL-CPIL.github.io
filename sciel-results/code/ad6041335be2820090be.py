"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import itertools
import math

def fit(X_train, y_train):
    """
    Symbolic regression via a rich feature basis + non-negative least squares
    with L1 regularization (Lasso-like) to select important terms.
    We use a broad set of candidate features including polynomials, trig,
    exp, log, abs, and interactions, then fit with a regularized approach.
    """
    X = np.atleast_2d(X_train)
    d = X.shape[1]
    n = X.shape[0]
    
    # Generate a rich feature basis
    def build_features(X):
        cols = [np.ones(len(X))]
        
        # Polynomial terms up to degree 3 (for 1D input, this is manageable)
        for j in range(d):
            for p in range(1, 4):
                cols.append(X[:, j] ** p)
        
        # Interactions for multi-dimensional (though task says 1D mostly)
        for i in range(d):
            for j in range(i, d):
                for p in range(1, 3):
                    for q in range(1, 3):
                        if i == j and p >= q:
                            continue
                        cols.append((X[:, i] ** p) * (X[:, j] ** q))
        
        # Trig functions
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
            cols.append(np.sin(0.5 * X[:, j]))
            cols.append(np.cos(0.5 * X[:, j]))
        
        # Exp and log (with safe handling)
        for j in range(d):
            cols.append(np.exp(X[:, j] / 3.0))
            cols.append(np.exp(-X[:, j] / 3.0))
            # Safe log: log(|x| + eps)
            cols.append(np.log(np.abs(X[:, j]) + 1e-6))
            # Safe log for positive range
            pos_mask = X[:, j] > 0
            if np.any(pos_mask):
                log_pos = np.zeros(len(X))
                log_pos[pos_mask] = np.log(X[:, j][pos_mask])
                cols.append(log_pos)
        
        # Abs and sign
        for j in range(d):
            cols.append(np.abs(X[:, j]))
            cols.append(np.sign(X[:, j]))
        
        # Rational-like terms
        for j in range(d):
            cols.append(1.0 / (np.abs(X[:, j]) + 1.0))
            cols.append(1.0 / (X[:, j] ** 2 + 1.0))
        
        # sqrt
        for j in range(d):
            cols.append(np.sqrt(np.abs(X[:, j])))
        
        F = np.column_stack(cols)
        return F
    
    F_train = build_features(X)
    
    # Handle any NaN or Inf
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features for better regularization behavior
    F_mean = F_train.mean(axis=0)
    F_std = F_train.std(axis=0)
    F_std[F_std == 0] = 1.0
    F_train_std = (F_train - F_mean) / F_std
    
    # Try multiple regularization strengths and pick best
    alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    best_w = None
    best_score = np.inf
    
    for alpha in alphas:
        try:
            if alpha == 0.0:
                # Ridge with very small lambda for stability
                w, *_ = np.linalg.lstsq(F_train_std, y_train, rcond=None)
            else:
                # Use scipy's lasso via linear_model if available, else use iterative
                # We'll use a simple coordinate descent Lasso
                w = _lasso_cd(F_train_std, y_train, alpha)
            
            # Evaluate on training data (as proxy since no test set)
            pred = F_train_std @ w
            mse = np.mean((pred - y_train) ** 2)
            var_y = np.var(y_train)
            if var_y > 1e-12:
                nrmse = mse / var_y
            else:
                nrmse = mse
            
            # Penalize complexity
            score = nrmse + alpha * np.sum(np.abs(w))
            
            if score < best_score:
                best_score = score
                best_w = w
        except Exception:
            continue
    
    if best_w is None:
        # Fallback to simple linear regression
        w, *_ = np.linalg.lstsq(F_train_std, y_train, rcond=None)
        best_w = w
    
    def predict(X):
        X = np.atleast_2d(X)
        F = build_features(X)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F_std = (F - F_mean) / F_std
        return F_std @ best_w
    
    return predict

def _lasso_cd(X, y, alpha, max_iter=1000, tol=1e-6):
    """Coordinate descent for Lasso regression."""
    n, p = X.shape
    w = np.zeros(p)
    
    # Precompute column norms
    col_norms = np.sum(X ** 2, axis=0)
    col_norms[col_norms == 0] = 1e-10
    
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            # Compute residual without j-th feature
            r = y - X @ w + X[:, j] * w[j]
            rho = np.dot(X[:, j], r) / col_norms[j]
            w[j] = _soft_threshold(rho, alpha) / col_norms[j]
        
        if np.max(np.abs(w - w_old)) < tol:
            break
    
    return w

def _soft_threshold(rho, alpha):
    if rho > alpha:
        return rho - alpha
    elif rho < -alpha:
        return rho + alpha
    else:
        return 0.0
# EVOLVE-BLOCK-END
