"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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

    # We will fit degrees 1 to 5.
    # The output requires 6 coefficients (powers 0 to 5).
    # Note: The task says "ascending powers of local slow state x".
    # So coeff[0] is c0, coeff[1] is c1, ..., coeff[5] is c5.
    
    best_score = np.inf
    best_coeff = np.zeros(6)
    
    # K-Fold Cross Validation setup
    k_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42) # Fixed seed for reproducibility in CV splits
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    # Regularization strengths to try
    # Log-spaced from 1e-2 to 1e4
    lambdas = np.logspace(-2, 4, 20)
    
    max_degree = 5
    
    # Pre-compute Vandermonde matrices for each degree?
    # Or compute on the fly? 
    # Computing V for degree 5 is O(N*5). It's cheap.
    
    # We iterate over degrees 1 to 5
    for d in range(1, max_degree + 1):
        if time.time() > deadline - 5:
            break
            
        # Build Vandermonde matrix for this degree d
        # Powers 0 to d
        V = np.vander(x_norm, N=d+1, increasing=True)
        
        # We need to evaluate the model. 
        # Ridge solution: beta = (V^T V + lambda I)^-1 V^T y
        # To speed up CV, we can precompute V^T V and V^T y for the full set?
        # No, CV requires training on subsets.
        
        # Efficient CV for Ridge:
        # We can use the "efficient" trick if we had a fixed lambda, 
        # but since we are trying many lambdas, let's just do direct solves.
        # N is likely small enough (< 100k) that this is fine within 110s.
        
        best_lambda_for_d = None
        best_cv_error_for_d = np.inf
        
        for lam in lambdas:
            if time.time() > deadline - 2:
                break
                
            cv_errors = []
            
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                V_train = V[train_idx]
                y_train = y[train_idx]
                V_val = V[val_idx]
                y_val = y[val_idx]
                
                # Solve Ridge
                # (V^T V + lam I) beta = V^T y
                A = V_train.T @ V_train
                A[np.diag_indices_from(A)] += lam
                b = V_train.T @ y_train
                
                try:
                    beta = solve(A, b, assume_a='pos')
                except:
                    # Fallback to lstsq if singular
                    try:
                        beta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                
                # Predict validation
                y_pred = V_val @ beta
                se = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(se)
            
            if cv_errors:
                mean_cv_error = np.mean(cv_errors)
                if mean_cv_error < best_cv_error_for_d:
                    best_cv_error_for_d = mean_cv_error
                    best_lambda_for_d = lam
        
        if best_lambda_for_d is not None:
            if best_cv_error_for_d < best_score:
                best_score = best_cv_error_for_d
                # Refit on full data with best lambda
                A = V.T @ V
                A[np.diag_indices_from(A)] += best_lambda_for_d
                b = V.T @ y
                
                try:
                    beta_norm = solve(A, b, assume_a='pos')
                except:
                    try:
                        beta_norm, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        beta_norm = np.zeros(d+1)
                
                # Map normalized coefficients back to original x scale?
                # Wait, the polynomial is in terms of x.
                # If we fit in x_norm = (x - mean) / std, the coefficients are for x_norm.
                # We need coefficients for x.
                # Let P(x_norm) = sum c_i x_norm^i.
                # x_norm = (x - mu) / sigma.
                # We need Q(x) = sum a_j x^j such that Q(x) = P((x-mu)/sigma).
                # This requires transforming the polynomial coefficients.
                
                # Alternatively, just fit in original x?
                # Vandermonde in original x can be ill-conditioned if x is large or has large variance.
                # Let's stick to normalization but transform back.
                
                # Transformation:
                # P(z) where z = (x-mu)/sigma
                # P(z) = c0 + c1 z + c2 z^2 + ...
                # We want a0 + a1 x + a2 x^2 + ...
                
                # This is a change of basis.
                # It's easier to just build the Vandermonde in original x if the range is reasonable.
                # Lorenz 96 states are typically O(1) to O(10). 
                # x_std is likely O(1).
                # Let's check if we can just use original x.
                # If we use original x, no transformation is needed.
                # Let's restart the loop using original x to avoid transformation complexity, 
                # but add a small regularization to handle condition number.
                # Actually, the current loop uses x_norm. 
                # Let's do the transformation properly.
                
                # P(z) = sum_{i=0}^d c_i z^i
                # z = (x - mu) / sigma = (1/sigma) x - (mu/sigma)
                # Let a = 1/sigma, b = -mu/sigma.
                # z = a x + b
                # P(a x + b) = sum_{i=0}^d c_i (a x + b)^i
                # (a x + b)^i = sum_{j=0}^i binom(i, j) (a x)^j b^{i-j}
                # = sum_{j=0}^i binom(i, j) a^j b^{i-j} x^j
                # So, a_j = sum_{i=j}^d c_i binom(i, j) a^j b^{i-j}
                
                a_val = 1.0 / x_std if x_std > 1e-10 else 1.0
                b_val = -x_mean / x_std if x_std > 1e-10 else 0.0
                
                beta_orig = np.zeros(6)
                for j in range(d + 1):
                    coeff_j = 0.0
                    for i in range(j, d + 1):
                        # binom(i, j)
                        binom_val = 1.0
                        if i > j:
                            # Calculate binomial coefficient
                            # i! / (j! (i-j)!)
                            # Use math.comb if available, or manual
                            try:
                                import math
                                binom_val = math.comb(i, j)
                            except:
                                binom_val = 1.0 # Fallback, should not happen for small i,j
                        term = beta_norm[i] * binom_val * (a_val ** j) * (b_val ** (i - j))
                        coeff_j += term
                    beta_orig[j] = coeff_j
                
                # Update best_coeff
                best_coeff = beta_orig
                # Ensure we store the degree used for potential debugging, though not required
                # best_degree = d
                
    # If no valid model was found (e.g. all failed), fallback to linear
    if np.all(best_coeff == 0):
        # Simple linear fit on original x
        A = np.vstack([x, np.ones_like(x)]).T
        try:
            res, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            best_coeff[0] = res[0]
            best_coeff[1] = res[1]
        except:
            pass

    return best_coeff
# EVOLVE-BLOCK-END
