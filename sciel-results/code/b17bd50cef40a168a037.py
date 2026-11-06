"""Improved program for the suite task; fits an odd-power polynomial closure exploiting antisymmetry."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 10:
        # Fallback to simple linear fit if data is too sparse
        try:
            # For tiny data, just fit linear (x^1 term)
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
        # Strategy: Exploit the physical antisymmetry of the unresolved forcing.
        # In the two-scale Lorenz-96 model, the fast-scale forcing is an odd function 
        # of the local slow state x (due to the symmetry of the fast dynamics and 
        # the linear coupling in dY). 
        # Therefore, we fit only odd powers: x^1, x^3, x^5.
        # The coefficients for even powers (x^0, x^2, x^4) are set to 0.
        
        # Construct design matrix for odd powers only
        # Columns: x, x^3, x^5
        A_odd = np.column_stack((x, x**3, x**5))
        
        # We will select the regularization parameter alpha using k-fold cross-validation.
        # Objective: minimize ||A_odd @ beta - y||^2 + alpha * ||beta||^2
        # Solution: beta = (A_odd^T A_odd + alpha * I)^-1 A_odd^T y
        
        # Precompute for speed? 
        # Since we split in CV, we need to compute per fold.
        # However, N is likely manageable. Let's use a small grid for alpha.
        
        # Define a grid of alpha values.
        # Regularization helps stabilize the high-order terms (x^5) which can be noisy.
        alphas = np.logspace(-6, 6, 25)
        
        best_alpha = 0.0
        best_cv_error = np.inf
        
        # K-Fold Cross Validation (K=5)
        K = 5
        n_samples = len(x)
        
        # Deterministic split for reproducibility and speed
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
                
                A_train = A_odd[train_idx]
                y_train = y[train_idx]
                A_val = A_odd[val_idx]
                y_val = y[val_idx]
                
                # Solve regularized least squares
                ATA_train = A_train.T @ A_train
                ATy_train = A_train.T @ y_train
                
                # Regularization matrix
                # Identity of size 3 (since we have 3 coefficients: c1, c3, c5)
                reg_matrix = ATA_train + alpha * np.eye(3)
                
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
            # Fallback to unregularized odd fit if running out of time
            try:
                beta, _, _, _ = np.linalg.lstsq(A_odd, y, rcond=None)
            except:
                beta = np.zeros(3)
            coeffs = np.zeros(6)
            coeffs[1] = beta[0]
            coeffs[3] = beta[1]
            coeffs[5] = beta[2]
            return coeffs
        
        # Final fit on all data with the best alpha
        if best_alpha == 0.0:
            beta, _, _, _ = np.linalg.lstsq(A_odd, y, rcond=None)
        else:
            ATA_full = A_odd.T @ A_odd
            ATy_full = A_odd.T @ y
            reg_matrix_full = ATA_full + best_alpha * np.eye(3)
            try:
                beta = np.linalg.solve(reg_matrix_full, ATy_full)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A_odd, y, rcond=None)[0]
                
        # Map 3 coefficients to 6 coefficients (ascending powers 0..5)
        coeffs = np.zeros(6)
        coeffs[1] = beta[0] # x^1
        coeffs[3] = beta[1] # x^3
        coeffs[5] = beta[2] # x^5
        
        # Ensure coefficients are finite
        if not np.all(np.isfinite(coeffs)):
            # Fallback to linear fit
            try:
                A_lin = x.reshape(-1, 1)
                coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
                coeffs = np.zeros(6)
                coeffs[1] = coeffs_lin[0]
                return coeffs
            except Exception:
                return np.zeros(6)
                
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
