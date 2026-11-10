"""Rewritten program: robust L1-regularized polynomial fit for unresolved tendency closure."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


def _solve_lasso(V, y, lam):
    """
    Solve Lasso problem: min 0.5/N * ||y - Vw||^2 + lam * ||w||_1
    Note: scipy's l1_minimize minimizes 0.5 * ||y - Vw||^2 + lam * ||w||_1.
    To match the 0.5/N factor, we can scale lam or V, y.
    Let's define the problem as min 0.5 * ||y - Vw||^2 + (lam * N) * ||w||_1 
    if we want the penalty to be lam * ||w||_1 in the normalized 0.5/N sense?
    Actually, let's just use the standard definition:
    min 0.5 * sum((y - Vw)^2) + lam * sum(|w|)
    The current code uses: min 0.5/N * ||y - Vw||^2 + lam * ||w||_1
    So the effective lambda in scipy's sense should be lam * N.
    """
    # scipy.optimize.l1_minimize minimizes: 0.5 * ||y - Vw||^2 + lam * ||w||_1
    # Our target: 0.5/N * ||y - Vw||^2 + lam * ||w||_1
    # Multiply our target by N: 0.5 * ||y - Vw||^2 + (lam * N) * ||w||_1
    # So we pass lam * N to l1_minimize.
    
    try:
        res = so.l1_minimize(V, y, lam * len(y), method='l1_ls')
        return res.x
    except Exception:
        # Fallback to simple OLS if solver fails
        try:
            return np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            return np.zeros(V.shape[1])


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses L1 regularization (Lasso) to select relevant polynomial terms.
    The regularization strength is selected via 5-fold cross-validation.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    x = x.ravel()
    y = y.ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Build full polynomial design matrix (degrees 0-5)
    V = np.vander(x, N=6, increasing=True)
    
    # Normalize V and y for numerical stability
    V_mean = V.mean(axis=0)
    V_std = V.std(axis=0)
    V_std[V_std < 1e-12] = 1.0
    V_norm = (V - V_mean) / V_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_norm = np.zeros_like(y)
        y_std_eff = 1.0
    else:
        y_norm = (y - y_mean) / y_std
        y_std_eff = y_std
    
    # Grid of L1 penalties to try
    # Use geometric spacing for better coverage
    log_lams = np.linspace(-8, 2, 100)
    lams = np.exp(log_lams)
    
    # 5-Fold Cross Validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42) # For deterministic CV split
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    best_lam = 0.0
    best_cv_mse = np.inf
    
    for lam in lams:
        cv_mses = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            V_train = V_norm[train_idx]
            y_train = y_norm[train_idx]
            V_val = V_norm[val_idx]
            y_val = y_norm[val_idx]
            
            # Solve Lasso on training fold
            w_fold = _solve_lasso(V_train, y_train, lam)
            
            # Evaluate on validation fold
            pred_val = V_val @ w_fold
            mse_val = np.mean((y_val - pred_val) ** 2)
            cv_mses.append(mse_val)
        
        avg_cv_mse = np.mean(cv_mses)
        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_lam = lam
    
    # Final fit on all data with best lam
    w_final = _solve_lasso(V_norm, y_norm, best_lam)
    
    # Convert back to original scale
    c = w_final / V_std
    bias = y_mean - y_std * np.dot(V_mean, c)
    
    c_orig = np.zeros(6)
    c_orig[0] = c[0] * y_std + bias
    c_orig[1:] = c[1:] * y_std
    
    # Final check
    if not np.all(np.isfinite(c_orig)):
        # Fallback to OLS
        try:
            c_ols = np.linalg.lstsq(V, y, rcond=None)[0]
            c_orig = c_ols
        except Exception:
            c_orig = np.zeros(6)
            
    return c_orig
# EVOLVE-BLOCK-END
