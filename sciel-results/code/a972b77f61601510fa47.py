"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Strategy:
    # 1. Construct the Vandermonde matrix for a degree-5 polynomial.
    # 2. Apply Tikhonov regularization (Ridge Regression) to the normal equations.
    #    The regularization parameter lambda is chosen adaptively based on the
    #    condition number of the design matrix to balance bias and variance.
    #    This avoids the bias of kernel regression towards training density and
    #    the instability of unregularized least squares on potentially noisy data.
    
    # Center and scale x to improve numerical stability of the Vandermonde matrix
    # This is crucial for high-degree polynomials where powers of x can vary wildly.
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
        
    x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for scaled x
    # Columns: 1, x_s, x_s^2, ..., x_s^5
    A_scaled = np.vander(x_scaled, 6, increasing=True)
    
    # Check time
    if time.time() - t_start > deadline - 5.0:
        # Fallback to simple mean if we are out of time (should not happen)
        return np.array([np.mean(y)] + [0.0]*5)
    
    # Calculate regularization parameter
    # We want to stabilize the inverse of A^T A.
    # Compute SVD of A_scaled to get singular values
    try:
        U, s, Vt = np.linalg.svd(A_scaled, full_matrices=False)
        s_max = s[0] if len(s) > 0 else 1.0
        s_min = s[-1] if len(s) > 0 else 1e-10
        cond = s_max / max(s_min, 1e-10)
        
        # Adaptive lambda: proportional to the largest singular value squared,
        # scaled by a factor that increases with condition number.
        # A common heuristic is lambda ~ s_max^2 * epsilon, but we want it strong enough
        # to regularize high-frequency noise.
        # Let's use a fraction of the largest eigenvalue of A^T A (which is s_max^2).
        # We'll start with a small fraction and increase if cond is high.
        base_lambda = 1e-4 * (s_max ** 2)
        if cond > 100:
             base_lambda *= 10.0
        if cond > 1000:
             base_lambda *= 10.0
             
        # Ensure lambda is not zero
        lambda_reg = max(base_lambda, 1e-10)
        
    except Exception:
        # If SVD fails, use a conservative default
        lambda_reg = 1e-2 * np.mean(y**2) if n > 0 else 1e-2

    # Solve regularized normal equations:
    # (A^T A + lambda * I) c = A^T y
    # Note: We regularize all coefficients. Some might argue against regularizing the intercept,
    # but for polynomial fits, regularizing all terms helps prevent wild swings.
    
    AtA = A_scaled.T @ A_scaled
    Aty = A_scaled.T @ y
    
    # Add regularization to the diagonal
    reg_matrix = AtA + lambda_reg * np.eye(6)
    
    try:
        # Solve for coefficients in the scaled space
        coeff_scaled = np.linalg.solve(reg_matrix, Aty)
        
        # Transform coefficients back to original x space
        # y = c0 + c1*x_s + ... + c5*x_s^5
        # x_s = (x - x_mean) / x_std
        # We need coefficients c_orig such that y = sum(c_orig_i * x^i)
        # This requires expanding the polynomial in terms of x.
        # Instead of symbolic expansion, we can simply compute the transformation matrix.
        # Let P be the matrix where P[i, j] is the coefficient of x^j in (x_s)^i.
        # Actually, it's easier to just use the scaled coefficients to evaluate,
        # but the task requires coefficients in ASCENDING powers of LOCAL SLOW STATE x.
        # So we MUST transform.
        
        # We want c_orig such that:
        # sum_{i=0}^5 c_orig_i x^i = sum_{j=0}^5 c_scaled_j ((x - mu)/sigma)^j
        # Let z = (x - mu)/sigma => x = sigma * z + mu
        # We substitute x = sigma*z + mu into the left side? No, we substitute z = (x-mu)/sigma into the right side.
        # RHS = sum_{j=0}^5 c_scaled_j * sum_{k=0}^j binomial(j, k) * (x-mu)^k * sigma^{-j}
        # This is messy. Let's use a different approach.
        
        # We can construct a transformation matrix T such that c_orig = T @ c_scaled.
        # Consider the basis functions:
        # phi_scaled_j(x) = ((x - mu)/sigma)^j
        # phi_orig_i(x) = x^i
        # We have phi_scaled_j(x) = sum_{i=0}^j T_{i,j} phi_orig_i(x)
        # And we want c_orig such that sum c_orig_i phi_orig_i(x) = sum c_scaled_j phi_scaled_j(x)
        # => sum c_orig_i phi_orig_i(x) = sum c_scaled_j (sum_{i=0}^j T_{i,j} phi_orig_i(x))
        # => c_orig_i = sum_{j=i}^5 c_scaled_j T_{i,j}
        
        # Let's compute T.
        # (x - mu)^k = sum_{i=0}^k binomial(k, i) x^i (-mu)^(k-i)
        # So ((x - mu)/sigma)^j = sigma^{-j} sum_{k=0}^j binomial(j, k) (x-mu)^k
        # = sigma^{-j} sum_{k=0}^j binomial(j, k) sum_{i=0}^k binomial(k, i) x^i (-mu)^(k-i)
        
        # This is computable but prone to error. 
        # Alternative: Since we only have 6 coefficients, we can just solve for c_orig
        # by matching the polynomial values at 6 distinct points? No, that's unstable.
        
        # Better: Use the fact that we can just evaluate the scaled polynomial at original x?
        # No, the output must be coefficients.
        
        # Let's do the transformation explicitly using a small loop.
        # T[i, j] is the coefficient of x^i in ((x-mu)/sigma)^j.
        T = np.zeros((6, 6))
        for j in range(6):
            # Expand ((x - mu)/sigma)^j
            # This is (1/sigma)^j * (x - mu)^j
            # (x - mu)^j = sum_{k=0}^j binomial(j, k) x^k (-mu)^(j-k)
            factor = (1.0 / x_std) ** j
            for k in range(j + 1):
                coeff_k = np.math.comb(j, k) * (-x_mean) ** (j - k)
                T[k, j] += factor * coeff_k
                
        c_orig = T.T @ c_scaled
        
        # Verify finiteness
        if not np.all(np.isfinite(c_orig)):
             # Fallback to unregularized least squares in original space if transformation fails
             A_orig = np.vander(x, 6, increasing=True)
             c_orig, _, _, _ = np.linalg.lstsq(A_orig, y, rcond=None)
             
    except Exception:
        # Fallback to simple linear fit on raw data if everything fails
        try:
            A_lin = np.vander(x, 2, increasing=True)
            coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            out = np.zeros(6)
            out[:2] = coeff_lin
            return out
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    return c_orig
# EVOLVE-BLOCK-END
