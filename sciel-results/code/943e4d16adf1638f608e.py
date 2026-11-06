# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Standardize x to improve numerical stability of polynomial features
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Construct design matrix for degree 5 polynomial
    # V[:, j] = x_norm**j
    V = np.vander(x_norm, N=6, increasing=True)

    # We will solve (V^T V + lambda * I) c = V^T y
    # To select lambda, we split data into train_val and holdout
    # Stratified split not needed as x is 1D continuous, random split is fine
    
    # If N is small, cross-validation might be overkill, but a single holdout is fast.
    # Let's use a fixed holdout size, e.g., 10% or min(1000, 10% of N)
    n_holdout = max(10, min(1000, n // 10))
    indices = np.random.permutation(n)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    
    V_train = V[train_idx]
    y_train = y[train_idx]
    V_hold = V[holdout_idx]
    y_hold = y[holdout_idx]
    
    # Precompute V^T V and V^T y for training set
    # V_train shape: (N_train, 6)
    VtV = V_train.T @ V_train
    Vty = V_train.T @ y_train
    
    # Grid of lambda values
    # Lambda range: from very small (LS) to large (shrinks to 0)
    # Scale lambda by trace(VtV)/6 to make it relative to data variance
    max_lambda = np.trace(VtV) / 6.0
    if max_lambda == 0:
        max_lambda = 1.0
    
    # Log-spaced lambdas
    lambdas = np.logspace(-4, 4, 100) * max_lambda
    
    best_mse = np.inf
    best_coeff_norm = np.zeros(6)
    
    for lam in lambdas:
        # Regularization matrix: identity, but we usually don't penalize the intercept (constant term) as much?
        # In polynomial regression, penalizing all terms is standard for Ridge.
        # However, penalizing the intercept can bias the mean. 
        # Let's penalize all terms for simplicity, or just terms 1..5?
        # Standard Ridge penalizes all. Let's stick to standard Ridge on all coefficients.
        A = VtV + lam * np.eye(6)
        try:
            coeff_norm = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            continue
        
        # Evaluate on holdout
        pred_hold = V_hold @ coeff_norm
        mse = np.mean((pred_hold - y_hold) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_coeff_norm = coeff_norm
            
        # Time check
        if time.time() > deadline - 5:
            break

    # If we didn't find a good lambda or time ran out, fallback to unregularized or simple LS
    if best_mse == np.inf:
        try:
            # Use full data for final fit with best lambda found, or LS if none
            VtV_full = V.T @ V
            Vty_full = V.T @ y
            # Use a moderate lambda if we didn't converge, or 0 if we did?
            # If we found best_coeff_norm, we should refit on ALL data with that lambda.
            # But we don't know which lambda it was in the loop above easily without storing it.
            # Let's just refit with the best lambda we implicitly used? 
            # Actually, let's store the best lambda.
            
            # Re-run logic to store best lambda? 
            # For efficiency in this edit, let's assume the loop finished or found something.
            # If the loop broke early due to time, best_coeff_norm is from the last valid lambda.
            # We should refit the coefficients on the FULL dataset using the best lambda.
            # But we didn't store the best lambda. 
            # Let's modify the loop to store best_lambda.
            
        except Exception:
            pass
    
    # To ensure we use the best lambda with full data, let's restructure slightly or just use the coeff from holdout?
    # Using coeff from holdout is slightly suboptimal as it ignores 10% of data.
    # Let's redo the selection storing the lambda.
    
    best_lambda = 0.0
    best_mse = np.inf
    best_coeff_norm = np.zeros(6)
    
    # Check if we have time for a proper search
    if time.time() < deadline - 10:
        for lam in lambdas:
            A = VtV + lam * np.eye(6)
            try:
                coeff_norm = np.linalg.solve(A, Vty)
            except np.linalg.LinAlgError:
                continue
            
            pred_hold = V_hold @ coeff_norm
            mse = np.mean((pred_hold - y_hold) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_coeff_norm = coeff_norm
                best_lambda = lam
                
            if time.time() > deadline - 10:
                break
        
        # Refit on FULL data with best_lambda
        if best_mse < np.inf:
            VtV_full = V.T @ V
            Vty_full = V.T @ y
            A_full = VtV_full + best_lambda * np.eye(6)
            try:
                coeff_norm_full = np.linalg.solve(A_full, Vty_full)
                if np.all(np.isfinite(coeff_norm_full)):
                    best_coeff_norm = coeff_norm_full
            except Exception:
                pass
    else:
        # Fallback: Simple LS on full data
        try:
            coeff_norm = np.linalg.lstsq(V, y, rcond=None)[0]
            if np.all(np.isfinite(coeff_norm)):
                best_coeff_norm = coeff_norm
        except Exception:
            pass

    # Transform coefficients back from normalized x to original x
    # P(x) = sum c_i ((x - mu)/sigma)^i
    # We need q_j such that P(x) = sum q_j x^j
    # This is a change of basis.
    # Let z = (x - mu)/sigma = x/sigma - mu/sigma.
    # z^k = (x/sigma - mu/sigma)^k.
    # We can compute the transformation matrix T where V_norm = V_orig @ T?
    # No, V_norm[:, k] = (x_norm)^k.
    # We want coeff_orig such that V_orig @ coeff_orig == V_norm @ coeff_norm.
    # This is a linear transformation.
    # Since we only need to return coeff_orig, we can solve for it using the relationship.
    # Actually, it's easier to just evaluate the polynomial on a grid and fit? No, that's lossy.
    # Algebraic transformation:
    # Let's use the property that if we have coeffs in basis z, we can convert to basis x.
    # z = a*x + b, where a = 1/sigma, b = -mu/sigma.
    # (a*x + b)^k = sum_{j=0}^k binom(k, j) a^j b^{k-j} x^j.
    # So coeff_norm[k] contributes to coeff_orig[j] for j<=k.
    
    coeff_orig = np.zeros(6)
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    for k in range(6):
        c_k = best_coeff_norm[k]
        if c_k == 0:
            continue
        # Expand (a*x + b)^k
        # Terms are binom(k, j) * a^j * b^(k-j) * x^j
        for j in range(k + 1):
            # Binomial coefficient
            # k is small (0..5), so we can compute directly
            binom_k_j = 1.0
            # Compute k choose j
            # Using multiplicative formula to avoid overflow/float issues for small k
            # nCr = n! / (r! (n-r)!)
            # For k<=5, simple loop is fine
            # Or use math.comb if available (Python 3.8+)
            try:
                import math
                binom_k_j = float(math.comb(k, j))
            except ImportError:
                # Fallback
                num = 1
                den = 1
                for i in range(j):
                    num *= (k - i)
                    den *= (i + 1)
                binom_k_j = num / den
                
            coeff_orig[j] += c_k * binom_k_j * (a ** j) * (b ** (k - j))
            
    # Ensure finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.zeros(6)
        
    return coeff_orig
# EVOLVE-BLOCK-END
