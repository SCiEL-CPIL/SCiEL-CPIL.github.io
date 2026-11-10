"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve
import math

def _ridge_cv_eigen(V, y, folds, alphas, k_folds):
    """
    Efficient ridge regression with CV using eigen-decomposition.
    For each fold, precompute V_train^T V_train eigen-decomposition once,
    then solve for all alphas in batch.
    
    Returns: best_alpha, best_cv_error, beta_for_best_alpha
    """
    n = V.shape[0]
    d = V.shape[1]
    
    # Precompute per-fold: V_train^T V_train, V_train^T y, V_val, y_val
    fold_data = []
    for i in range(k_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        
        V_train = V[train_idx]
        y_train = y[train_idx]
        V_val = V[val_idx]
        y_val = y[val_idx]
        
        G = V_train.T @ V_train  # (d, d)
        g = V_train.T @ y_train  # (d,)
        
        # Eigen-decompose G
        try:
            eigvals, eigvecs = np.linalg.eigh(G)
        except:
            eigvals = np.ones(d)
            eigvecs = np.eye(d)
        
        # Clip small eigenvalues for numerical stability
        eigvals = np.maximum(eigvals, 1e-12)
        
        fold_data.append((eigvals, eigvecs, g, V_val, y_val))
    
    best_alpha = alphas[0]
    best_cv_error = np.inf
    best_beta = np.zeros(d)
    
    for alpha in alphas:
        cv_errors = []
        
        for (eigvals, eigvecs, g, V_val, y_val) in fold_data:
            # beta = V^T (V V^T + alpha I)^{-1} y
            # Using eigen: G = Q diag(w) Q^T
            # (G + alpha I)^{-1} = Q diag(1/(w+alpha)) Q^T
            # beta = Q diag(1/(w+alpha)) Q^T g
            
            denom = eigvals + alpha
            beta = eigvecs @ ((eigvecs.T @ g) / denom)
            
            y_pred = V_val @ beta
            se = np.mean((y_val - y_pred) ** 2)
            cv_errors.append(se)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_alpha = alpha
            # Recompute beta for full data later; store for now
            # Actually we need beta from full data refit, not from CV fold
            # We'll refit after finding best_alpha
    
    return best_alpha, best_cv_error

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n_samples = x.shape[0]
    
    # Handle edge cases
    if n_samples == 0:
        return np.zeros(6)
    
    # If very few samples, just do linear fit to avoid instability
    if n_samples < 10:
        coeff = np.zeros(6)
        if n_samples >= 2:
            # Simple linear fit
            A = np.vstack([x, np.ones_like(x)]).T
            try:
                res, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                coeff[0] = res[0]
                coeff[1] = res[1]
            except:
                pass
        return coeff

    # Normalize x to improve numerical stability of Vandermonde matrix
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_norm = x - x_mean
    else:
        x_norm = (x - x_mean) / x_std

    # K-Fold Cross Validation setup
    k_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)  # Fixed seed for reproducibility in CV splits
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    # Regularization strengths to try
    lambdas = np.logspace(-2, 4, 30)
    
    max_degree = 5
    
    best_score = np.inf
    best_coeff = np.zeros(6)
    best_degree_used = 0
    
    # Iterate over degrees 1 to 5
    for d in range(1, max_degree + 1):
        if time.time() > deadline - 10:
            break
            
        # Build Vandermonde matrix for this degree d (powers 0 to d)
        V = np.vander(x_norm, N=d+1, increasing=True)
        
        # Efficient CV using eigen-decomposition
        best_lambda_for_d, best_cv_error_for_d = _ridge_cv_eigen(
            V, y, folds, lambdas, k_folds
        )
        
        if best_cv_error_for_d < best_score:
            best_score = best_cv_error_for_d
            best_degree_used = d
            best_lambda = best_lambda_for_d
            
            # Refit on full data with best lambda
            A = V.T @ V
            A[np.diag_indices_from(A)] += best_lambda
            b = V.T @ y
            
            try:
                beta_norm = solve(A, b, assume_a='pos')
            except:
                try:
                    beta_norm, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    beta_norm = np.zeros(d+1)
            
            # Transform from normalized x to original x
            # P(z) = sum_{i=0}^d c_i z^i where z = (x - mu) / sigma
            # Q(x) = P((x-mu)/sigma) = sum_{j=0}^d a_j x^j
            # a_j = sum_{i=j}^d c_i * binom(i,j) * (1/sigma)^j * (-mu/sigma)^(i-j)
            
            a_val = 1.0 / x_std if x_std > 1e-10 else 1.0
            b_val = -x_mean / x_std if x_std > 1e-10 else 0.0
            
            beta_orig = np.zeros(6)
            for j in range(d + 1):
                coeff_j = 0.0
                for i in range(j, d + 1):
                    binom_val = math.comb(i, j)
                    term = beta_norm[i] * binom_val * (a_val ** j) * (b_val ** (i - j))
                    coeff_j += term
                beta_orig[j] = coeff_j
            
            best_coeff = beta_orig
    
    # If no valid model was found, fallback to linear
    if np.all(best_coeff == 0):
        A = np.vstack([x, np.ones_like(x)]).T
        try:
            res, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            best_coeff[0] = res[0]
            best_coeff[1] = res[1]
        except:
            pass

    return best_coeff
# EVOLVE-BLOCK-END
