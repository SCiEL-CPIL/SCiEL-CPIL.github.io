"""Climate dynamics: unresolved-tendency closure via regularized polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved subgrid tendency.
    
    Uses ridge regression with model selection via k-fold cross-validation
    on the training set to choose optimal regularization strength.
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    
    # Build Vandermonde matrix for degree-5 polynomial
    # Columns: [1, x, x^2, x^3, x^4, x^5]
    V = np.vander(x, N=6, increasing=True)
    
    # Check for degenerate cases
    if n < 6:
        # Not enough data for degree 5; use what we can
        # Fall back to linear fit
        if n >= 2:
            coeffs = np.zeros(6)
            coeffs[:2] = np.polyfit(x, y, 1)[::-1]  # polyfit returns descending
            return coeffs
        else:
            return np.zeros(6)
    
    # Try a range of regularization strengths
    # lambda controls ridge penalty: min ||V@c - y||^2 + lambda*||c||^2
    # We'll use k-fold CV to pick the best lambda
    
    # Generate candidate lambdas (log-spaced)
    # The scale of the penalty depends on the data; normalize V columns first
    V_mean = V.mean(axis=0)
    V_std = V.std(axis=0)
    V_std[V_std < 1e-12] = 1.0
    V_norm = (V - V_mean) / V_std
    
    # Candidate regularization strengths
    lambda_candidates = np.logspace(-8, 2, 25)
    
    # K-fold cross-validation
    n_folds = min(5, n)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds.append(indices[start:end])
    
    best_lambda = 0.0
    best_cv_mse = np.inf
    
    for lam in lambda_candidates:
        if time.time() > DEADLINE:
            break
        cv_errors = []
        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
            
            V_train = V_norm[train_idx]
            y_train = y[train_idx]
            V_val = V_norm[val_idx]
            y_val = y[val_idx]
            
            # Ridge solve: (V^T V + lambda*I) c = V^T y
            n_params = 6
            A = V_train.T @ V_train + lam * np.eye(n_params)
            b = V_train.T @ y_train
            
            try:
                c_norm = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                c_norm = np.linalg.lstsq(A, b, rcond=None)[0]
            
            pred_val = V_val @ c_norm
            mse = np.mean((pred_val - y_val) ** 2)
            cv_errors.append(mse)
        
        mean_cv_mse = np.mean(cv_errors)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_lambda = lam
    
    # Final fit with best lambda on all training data (using normalized V)
    A = V_norm.T @ V_norm + best_lambda * np.eye(6)
    b = V_norm.T @ y
    
    try:
        c_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c_norm = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Convert back from normalized space to original polynomial coefficients
    # V_norm @ c_norm = V @ c_original
    # (V - V_mean)/V_std * c_norm = V @ c_original
    # V @ c_original = V_norm @ c_norm
    # c_original = V^+ @ (V_norm @ c_norm) where V^+ is pseudo-inverse
    # More directly: V @ c_orig = V_norm @ c_norm
    # c_orig = pinv(V) @ (V_norm @ c_norm)
    
    # But actually, the polynomial is: sum(c_i * x^i)
    # V is the Vandermonde with increasing powers
    # V @ c_orig gives predictions
    # We need c_orig such that V @ c_orig = V_norm @ c_norm
    # c_orig = pinv(V) @ (V_norm @ c_norm)
    
    try:
        c_orig = np.linalg.lstsq(V, V_norm @ c_norm, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback: use the normalized coefficients directly (not correct but safe)
        c_orig = c_norm
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(c_orig)):
        # Fall back to simple linear fit
        c_orig = np.zeros(6)
        if n >= 2:
            c_orig[:2] = np.polyfit(x, y, 1)[::-1]
    
    return c_orig
# EVOLVE-BLOCK-END
