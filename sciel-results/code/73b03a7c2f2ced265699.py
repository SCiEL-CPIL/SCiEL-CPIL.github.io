"""Symbolic regression via SINDy-style sparse regression with rich basis library."""
import numpy as np
import time
from scipy.optimize import lasso_lars

def _build_library(X, d, max_poly=3):
    """
    Build a library of candidate basis functions:
    - Polynomials up to degree max_poly (all monomials)
    - Trigonometric functions (sin, cos) of each input
    - Exponential functions of each input
    - Interactions of trig/exp with inputs
    - Constant
    """
    n = X.shape[0]
    cols = [np.ones(n)]
    
    # Polynomial monomials up to degree max_poly
    for deg in range(1, max_poly + 1):
        # Generate all monomials of degree 'deg'
        from itertools import combinations_with_replacement
        for combo in combinations_with_replacement(range(d), deg):
            term = np.ones(n)
            for idx in combo:
                term *= X[:, idx]
            cols.append(term)
    
    # Trigonometric functions of each input
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.sin(2 * X[:, j]))
        cols.append(np.cos(2 * X[:, j]))
    
    # Exponential functions of each input (scaled to avoid overflow)
    for j in range(d):
        xj = X[:, j]
        # Use a moderate scaling to keep values reasonable
        cols.append(np.exp(0.1 * xj))
        cols.append(np.exp(-0.1 * xj))
    
    # Interactions: sin/cos with other inputs
    for i in range(d):
        for j in range(d):
            if i != j:
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.cos(X[:, j]))
    
    # Product of two inputs with trig
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j] * np.sin(X[:, i]))
            cols.append(X[:, i] * X[:, j] * np.cos(X[:, j]))
    
    return np.column_stack(cols)


def _sparse_regression(L, y, alpha_candidates=None, time_budget=10.0):
    """
    Perform sparse regression using LARS/Lasso path.
    Returns the coefficient vector that minimizes validation error
    on a held-out split, using the Lasso path for stability.
    """
    n = L.shape[0]
    
    # Standardize L columns for numerical stability
    L_mean = L.mean(axis=0)
    L_std = L.std(axis=0)
    L_std[L_std < 1e-10] = 1.0
    L_norm = (L - L_mean) / L_std
    
    # Hold-out split for model selection
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_val = max(20, n // 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    L_train = L_norm[train_idx]
    L_val = L_norm[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    # Compute Lasso path
    try:
        # Use lasso_lars to get the full path
        alphas = np.logspace(-4, 2, 50)
        coefs_path = np.zeros((len(alphas), L_train.shape[1]))
        for i, alpha in enumerate(alphas):
            coefs, _, _, _, _, _, _, _ = lasso_lars(
                L_train, y_train,
                alpha=alpha,
                fit_intercept=True,
                return_path=False,
                n_nonzero_coefs=None,
                max_iter=10000,
                tol=1e-8
            )
            coefs_path[i] = coefs
    except Exception:
        # Fallback to simple ridge
        L_aug = np.column_stack([np.ones(L_train.shape[0]), L_train])
        w, *_ = np.linalg.lstsq(L_aug, y_train, rcond=None)
        L_val_aug = np.column_stack([np.ones(L_val.shape[0]), L_val])
        pred_val = L_val_aug @ w
        mse_val = np.mean((pred_val - y_val) ** 2)
        return w

    # Evaluate each alpha on validation set
    best_mse = np.inf
    best_coefs = coefs_path[0]
    for i, alpha in enumerate(alphas):
        L_val_aug = np.column_stack([np.ones(L_val.shape[0]), L_val])
        # lasso_lars with fit_intercept=True returns coefficients for the predictors only
        # We need to handle intercept separately
        # Actually, let's re-do this more carefully
        pass
    
    # Simpler approach: use the full Lasso path and evaluate
    best_mse = np.inf
    best_coefs = np.zeros(L_train.shape[1])
    best_intercept = 0.0
    
    for i, alpha in enumerate(alphas):
        try:
            coefs, _, _, _, _, _, _, _ = lasso_lars(
                L_train, y_train,
                alpha=alpha,
                fit_intercept=True,
                return_path=False,
                n_nonzero_coefs=None,
                max_iter=10000,
                tol=1e-8
            )
            # Predict on validation
            pred_val = L_val @ coefs
            # Need intercept - lasso_lars doesn't return it directly with this API
            # Let's compute it from training data
            residual = y_train - L_train @ coefs
            intercept = residual.mean()
            pred_val = L_val @ coefs + intercept
            mse_val = np.mean((pred_val - y_val) ** 2)
            if mse_val < best_mse:
                best_mse = mse_val
                best_coefs = coefs.copy()
                best_intercept = intercept
        except Exception:
            continue
    
    # Final fit on all data with the chosen alpha
    # Find which alpha gave best result
    # Re-fit on all data
    L_all_norm = L_norm
    # Determine alpha from best_coefs sparsity
    n_nonzero = np.sum(np.abs(best_coefs) > 1e-10)
    # Use a simple approach: refit with the same alpha
    # For now, use the coefs as-is (they were fit on training subset)
    # Better: refit on all data with the selected alpha
    
    # Find the alpha that corresponds to best_coefs
    best_alpha = alphas[0]
    min_diff = np.inf
    for i, alpha in enumerate(alphas):
        try:
            coefs_i, _, _, _, _, _, _, _ = lasso_lars(
                L_train, y_train,
                alpha=alpha,
                fit_intercept=True,
                return_path=False,
                n_nonzero_coefs=None,
                max_iter=10000,
                tol=1e-8
            )
            diff = np.sum((coefs_i - best_coefs) ** 2)
            if diff < min_diff:
                min_diff = diff
                best_alpha = alpha
        except:
            pass
    
    # Refit on all data with best_alpha
    try:
        coefs_all, _, _, _, _, _, _, _ = lasso_lars(
            L_all_norm, y,
            alpha=best_alpha,
            fit_intercept=True,
            return_path=False,
            n_nonzero_coefs=None,
            max_iter=10000,
            tol=1e-8
        )
        residual = y - L_all_norm @ coefs_all
        intercept_all = residual.mean()
    except Exception:
        coefs_all = best_coefs
        intercept_all = best_intercept
    
    return coefs_all, intercept_all, L_mean, L_std


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for 10 equations
    
    d = X_train.shape[1]
    
    # Build library
    L = _build_library(X_train, d, max_poly=3)
    
    # Ensure no NaN/Inf in library
    L = np.nan_to_num(L, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple linear
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(X_train.shape[0]), X_train]), y_train, rcond=None)
        return lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
    
    # Sparse regression
    try:
        result = _sparse_regression(L, y_train, time_budget=15.0)
        if len(result) == 4:
            coefs, intercept, L_mean, L_std = result
        else:
            # Fallback
            w, *_ = np.linalg.lstsq(np.column_stack([np.ones(X_train.shape[0]), X_train]), y_train, rcond=None)
            return lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
    except Exception:
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(X_train.shape[0]), X_train]), y_train, rcond=None)
        return lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
    
    # Store for predict
    L_shape = L.shape
    
    def predict(X):
        # Rebuild library for new X
        L_new = _build_library(X, d, max_poly=3)
        L_new = np.nan_to_num(L_new, nan=0.0, posinf=1e6, neginf=-1e6)
        # Standardize same way
        L_new_norm = (L_new - L_mean) / L_std
        pred = L_new_norm @ coefs + intercept
        return pred
    
    return predict
