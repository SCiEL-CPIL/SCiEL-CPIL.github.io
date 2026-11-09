"""Symbolic regression via optimized composite kernel ridge regression."""
import numpy as np
import time
from scipy.optimize import minimize

def _rbf_kernel(X1, X2, ls):
    """RBF kernel with independent length scales."""
    if np.isscalar(ls):
        ls = np.full(X1.shape[1], ls)
    X1_s = X1 / ls
    X2_s = X2 / ls
    sq_dists = np.sum(X1_s**2, axis=1)[:, None] + np.sum(X2_s**2, axis=1)[None, :] - 2 * (X1_s @ X2_s.T)
    np.maximum(sq_dists, 0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)

def _poly_kernel(X1, X2, degree, coef0):
    """Polynomial kernel: (coef0 + <x1, x2>)^degree."""
    return (coef0 + X1 @ X2.T) ** degree

def _periodic_kernel(X1, X2, period):
    """Periodic kernel based on Euclidean distance."""
    dists = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
    dists = np.maximum(dists, 0)
    return np.exp(-2.0 * np.sin(np.pi * dists / period)**2)

def _compute_composite_kernel(X1, X2, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period):
    """Compute composite kernel: rbf_var * RBF + poly_var * Poly + per_var * Periodic."""
    K = rbf_var * _rbf_kernel(X1, X2, rbf_ls)
    if poly_var > 1e-10:
        K = K + poly_var * _poly_kernel(X1, X2, poly_deg, 1.0)
    if per_var > 1e-10:
        K = K + per_var * _periodic_kernel(X1, X2, per_period)
    return K

def _kernel_ridge_solve(K, y, alpha):
    """Solve (K + alpha*I)^-1 y with fallbacks."""
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        try:
            return np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, y, rcond=None)[0]

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 290.0
    
    n, d = X_train.shape
    
    # Standardize inputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    # Standardize outputs
    y_mean = y_train.mean()
    y_std_val = float(y_train.std()) if y_train.std() > 1e-8 else 1.0
    y_norm = (y_train - y_mean) / y_std_val
    
    # Precompute LOOCV efficient solver
    # We will optimize hyperparameters using LOOCV which is O(n^2) per eval after O(n^3) decomposition.
    # But decomposition depends on hyperparams. So each eval is O(n^3).
    # With n=200, n^3 = 8e6. 290s / (8e6 * cost_per_op) allows for many evals.
    # Let's aim for ~100-200 evals in random search + local refinement.
    
    def loocv_score(rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha):
        K_train = _compute_composite_kernel(X_norm, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        K_reg = K_train + alpha * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K_reg)
            # alpha_vec = K_reg^-1 y
            # We need diag(K_reg^-1)
            # K_reg^-1 = L^-T L^-1
            # diag(K_reg^-1) = 1 / diag(L)^2
            diag_L = np.diag(L)
            K_ii = np.diag(K_train)
            
            # LOOCV prediction: y_hat_i = y_i - K_ii / (K_reg^-1)_{ii}
            # (K_reg^-1)_{ii} = 1 / diag_L[i]^2
            pred_i = y_norm - K_ii / (diag_L**2)
            errors = y_norm - pred_i
            mse = np.mean(errors**2)
            return mse
        except np.linalg.LinAlgError:
            return np.inf

    best_score = np.inf
    best_params = None
    
    # Random Search Phase
    # Hyperparameter ranges
    # rbf_ls: 0.1 to 10 per feature
    # rbf_var: 0.1 to 100
    # poly_deg: 1, 2, 3, 4
    # poly_var: 0.01 to 10
    # per_var: 0.01 to 10
    # per_period: 1 to 30
    # alpha: 1e-4 to 100
    
    rng = np.random.RandomState(42)
    
    n_random_evals = 200
    for i in range(n_random_evals):
        if time.time() > deadline - 20:
            break
            
        ls = 10 ** rng.uniform(-1, 1, size=d)
        rbf_var = 10 ** rng.uniform(-1, 2)
        poly_deg = int(rng.choice([1, 2, 3, 4]))
        poly_var = 10 ** rng.uniform(-2, 1)
        per_var = 10 ** rng.uniform(-2, 1)
        per_period = 10 ** rng.uniform(0, 1.5)
        alpha = 10 ** rng.uniform(-4, 2)
        
        try:
            score = loocv_score(ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha)
            if score < best_score:
                best_score = score
                best_params = (ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha)
        except Exception:
            continue
    
    if best_params is None:
        best_params = (np.ones(d), 1.0, 2, 1.0, 1.0, 5.0, 1.0)
    
    # Local Refinement Phase
    # Use Nelder-Mead or BFGS on the continuous parameters
    # Discrete poly_deg is handled by trying each degree in the neighborhood of the best found.
    
    best_ls, best_rbf_var, best_poly_deg, best_poly_var, best_per_var, best_per_period, best_alpha = best_params
    
    # Define objective for continuous params
    def objective_continuous(x):
        ls = np.exp(x[:d])
        rbf_var = np.exp(x[d])
        poly_var = np.exp(x[d+1])
        per_var = np.exp(x[d+2])
        per_period = np.exp(x[d+3])
        alpha = np.exp(x[d+4])
        
        # Clamp to reasonable bounds to prevent overflow/underflow
        ls = np.clip(ls, 0.01, 100.0)
        rbf_var = np.clip(rbf_var, 0.01, 1000.0)
        poly_var = np.clip(poly_var, 0.001, 100.0)
        per_var = np.clip(per_var, 0.001, 100.0)
        per_period = np.clip(per_period, 0.5, 100.0)
        alpha = np.clip(alpha, 1e-6, 1000.0)
        
        # Keep poly_deg fixed to best_poly_deg for this optimization
        try:
            score = loocv_score(ls, rbf_var, best_poly_deg, poly_var, per_var, per_period, alpha)
            return score
        except Exception:
            return np.inf
    
    # Initial x in log space
    x0 = np.concatenate([
        np.log(best_ls),
        [np.log(best_rbf_var)],
        [np.log(best_poly_var)],
        [np.log(best_per_var)],
        [np.log(best_per_period)],
        [np.log(best_alpha)]
    ])
    
    bounds = [(-4, 4)] * d + [(-4, 4), (-4, 4), (-4, 4), (-4, 4), (-4, 4)]
    
    # Try a few different poly_degrees around the best one found
    candidate_degrees = sorted(set([best_poly_deg - 1, best_poly_deg, best_poly_deg + 1, 1, 2, 3, 4]))
    candidate_degrees = [deg for deg in candidate_degrees if 1 <= deg <= 5]
    
    best_global_params = best_params
    best_global_score = best_score
    
    for deg in candidate_degrees:
        if time.time() > deadline - 10:
            break
        
        # Reset best_poly_deg for this optimization run
        current_best_deg = deg
        
        # Re-initialize x0 slightly if needed, or just use the previous best as start
        # It's often good to start from the best known continuous params
        x_start = x0.copy()
        
        try:
            res = minimize(objective_continuous, x_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if res.fun < best_global_score:
                ls = np.exp(res.x[:d])
                rbf_var = np.exp(res.x[d])
                poly_var = np.exp(res.x[d+1])
                per_var = np.exp(res.x[d+2])
                per_period = np.exp(res.x[d+3])
                alpha = np.exp(res.x[d+4])
                
                ls = np.clip(ls, 0.01, 100.0)
                rbf_var = np.clip(rbf_var, 0.01, 1000.0)
                poly_var = np.clip(poly_var, 0.001, 100.0)
                per_var = np.clip(per_var, 0.001, 100.0)
                per_period = np.clip(per_period, 0.5, 100.0)
                alpha = np.clip(alpha, 1e-6, 1000.0)
                
                best_global_score = res.fun
                best_global_params = (ls, rbf_var, deg, poly_var, per_var, per_period, alpha)
                x0 = res.x # Update start point for next degree
        except Exception:
            pass
    
    # Final fit with best params
    rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha = best_global_params
    
    K_final = _compute_composite_kernel(X_norm, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
    coeffs_final = _kernel_ridge_solve(K_final, y_norm, alpha)
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_test_norm = (X - X_mean) / X_std
        K_test = _compute_composite_kernel(X_test_norm, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        preds = K_test @ coeffs_final
        preds = preds * y_std_val + y_mean
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds
    
    return predict
