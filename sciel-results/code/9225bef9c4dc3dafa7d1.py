"""Rewritten program: robust L1-regularized polynomial fit for unresolved tendency closure."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


def _fista_lasso(V, y, lam, max_iter=2000, tol=1e-8):
    """
    Solve Lasso problem: min 0.5/N * ||y - Vw||^2 + lam * ||w||_1
    using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
    """
    n, d = V.shape
    # Lipschitz constant L for the gradient of the smooth part
    # L = max eigenvalue of (V^T V) / n
    try:
        VtV = V.T @ V
        eigvals = np.linalg.eigvalsh(VtV)
        L = max(eigvals.max() / n, 1e-10)
    except np.linalg.LinAlgError:
        L = np.max(np.sum(V**2, axis=0)) / n
    
    w = np.zeros(d)
    t = 1.0
    
    for _ in range(max_iter):
        # Gradient step
        grad = (V.T @ (V @ w - y)) / n
        w_new = w - grad / L
        
        # Proximal operator (soft thresholding)
        thresh = lam / L
        w_shrink = np.sign(w_new) * np.maximum(np.abs(w_new) - thresh, 0.0)
        
        # Check convergence
        if np.linalg.norm(w_shrink - w) < tol * (1 + np.linalg.norm(w)):
            w = w_shrink
            break
        
        # FISTA update
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        w = w_shrink + ((t - 1) / t_new) * (w_shrink - w)
        t = t_new
    
    return w


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses L1 regularization (Lasso) to select relevant polynomial terms.
    The regularization strength is selected via cross-validation.
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
    
    # Grid of L1 penalties to try in log space
    log_lams = np.linspace(-8, 1, 30)
    lams = np.exp(log_lams)
    
    # 5-Fold Cross Validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    best_lam = 0.0
    best_cv_mse = np.inf
    
    for lam in lams:
        cv_mses = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            # Solve Lasso on training fold
            w_fold = _fista_lasso(V_train, y_train, lam)
            
            # Evaluate on validation fold
            pred_val = V_val @ w_fold
            mse_val = np.mean((y_val - pred_val) ** 2)
            cv_mses.append(mse_val)
        
        avg_cv_mse = np.mean(cv_mses)
        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_lam = lam
    
    # Final fit on all data with best lam
    w_final = _fista_lasso(V, y, best_lam)
    
    # Verify solution is finite
    if not np.all(np.isfinite(w_final)):
        # Fallback to OLS
        try:
            w_final = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            w_final = np.zeros(6)
    
    # Additional safety: if Lasso gave a worse fit than OLS, use OLS
    try:
        w_ols = np.linalg.lstsq(V, y, rcond=None)[0]
        mse_lasso = np.mean((y - V @ w_final) ** 2)
        mse_ols = np.mean((y - V @ w_ols) ** 2)
        if mse_ols < mse_lasso * 0.95:  # Use OLS if it's significantly better
            w_final = w_ols
    except Exception:
        pass
    
    return w_final
# EVOLVE-BLOCK-END
