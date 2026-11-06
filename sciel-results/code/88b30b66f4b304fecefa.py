"""Improved program for the suite task; fits a regularized degree-five polynomial closure via cross-validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 10:
        # Fallback to simple linear fit if data is too sparse for CV
        try:
            A_lin = x.reshape(-1, 1)
            coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            coeffs = np.zeros(6)
            coeffs[1] = coeffs_lin[0]
            return coeffs
        except Exception:
            return np.zeros(6)
            
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # Construct the full Vandermonde design matrix for powers 0 to 5
        # Columns: 1, x, x^2, x^3, x^4, x^5
        # Using np.vander with increasing=True gives [1, x, x^2, ...]
        A = np.vander(x, N=6, increasing=True)
        
        # We will select the regularization parameter alpha (lambda) using k-fold cross-validation.
        # The objective is to minimize ||A @ beta - y||^2 + alpha * ||beta||^2.
        # The solution is beta = (A^T A + alpha * I)^-1 A^T y.
        
        # Precompute A^T A and A^T y for speed in the loop
        # Note: A is (N, 6), A^T A is (6, 6)
        ATA = A.T @ A
        ATy = A.T @ y
        
        # Define a grid of alpha values to search.
        # We span several orders of magnitude.
        # Small alpha -> OLS, Large alpha -> shrinks towards 0.
        alphas = np.logspace(-4, 4, 40)
        
        best_alpha = 0.0
        best_cv_error = np.inf
        
        # K-Fold Cross Validation (K=5)
        K = 5
        n_samples = len(x)
        
        # Create folds. We don't need to shuffle if the data is already mixed, 
        # but for safety we can create indices. 
        # However, shuffling adds overhead. Given the 120s limit and N is likely not huge,
        # simple contiguous folds or a fixed permutation is fine.
        # Let's use a fixed permutation to avoid RNG overhead and ensure reproducibility.
        indices = np.arange(n_samples)
        # Simple deterministic shuffle equivalent for CV split stability
        # np.random.seed is bad for reproducibility across runs if not managed, 
        # but we can just use the first K/5, next K/5, etc. 
        # Actually, random splitting is better for unbiased CV. 
        # Let's use np.random.default_rng with a fixed seed for the split logic only.
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_samples)
        folds = np.array_split(perm, K)
        
        # Check time periodically
        for i, alpha in enumerate(alphas):
            if time.time() - start_time > deadline - 5.0:
                break
                
            cv_errors = []
            for k in range(K):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
                
                # Extract train/val data
                A_train = A[train_idx]
                y_train = y[train_idx]
                A_val = A[val_idx]
                y_val = y[val_idx]
                
                # Solve regularized least squares for this alpha
                # (A_train^T A_train + alpha * I) @ beta = A_train^T y_train
                # We can use the precomputed ATA only if we don't split, but here we split.
                # So we must compute ATA_train.
                ATA_train = A_train.T @ A_train
                ATy_train = A_train.T @ y_train
                
                # Add regularization
                reg_matrix = ATA_train + alpha * np.eye(6)
                
                try:
                    beta = np.linalg.solve(reg_matrix, ATy_train)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(reg_matrix, ATy_train, rcond=None)[0]
                
                # Compute validation error
                y_pred = A_val @ beta
                error = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(error)
            
            mean_cv_error = np.mean(cv_errors)
            
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                best_alpha = alpha
        
        # Check time before final fit
        if time.time() - start_time > deadline - 2.0:
            # Fallback to OLS if running out of time
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return coeffs
        
        # Final fit on all data with the best alpha
        if best_alpha == 0.0:
             coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        else:
            # (A^T A + alpha * I) @ beta = A^T y
            reg_matrix = ATA + best_alpha * np.eye(6)
            try:
                coeffs = np.linalg.solve(reg_matrix, ATy)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
                
        # Ensure coefficients are finite
        if not np.all(np.isfinite(coeffs)):
            # Fallback to linear fit
            A_lin = x.reshape(-1, 1)
            coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            coeffs = np.zeros(6)
            coeffs[1] = coeffs_lin[0]
            return coeffs
            
        return coeffs
        
    except Exception:
        # Fallback to linear fit in case of any errors
        try:
            A_lin = x.reshape(-1, 1)
            coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            coeffs = np.zeros(6)
            coeffs[1] = coeffs_lin[0]
            return coeffs
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
