"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    using Bayesian Ridge Regression in the Chebyshev basis with exponentially
    decaying priors on the higher-order coefficients.
    
    Strategy:
    1. Map x to Chebyshev domain [-1, 1].
    2. Construct Chebyshev design matrix.
    3. Solve for Chebyshev coefficients using weighted least squares where
       higher-order coefficients are penalized more heavily (exponential decay).
       This acts as a structural smoother, reducing overfitting to noise in
       high-order modes which often hurts generalization to unseen regimes.
    4. Convert to monomial basis.
    """
    start_time = time.time()
    DEADLINE = 110.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        c0 = np.mean(y)
        return np.array([c0, 0, 0, 0, 0, 0])
    
    A = x_range / 2.0
    B = (x_max + x_min) / 2.0
    
    t = (x - B) / A
    t = np.clip(t, -1.0, 1.0)

    N = len(t)
    T = np.zeros((N, 6))
    T[:, 0] = 1.0
    if N > 0:
        T[:, 1] = t
        for k in range(2, 6):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Structural Change: Weighted Least Squares with exponential prior decay
    # We minimize ||T @ c - y||^2 + sum (c_k / tau_k)^2
    # This is equivalent to minimizing ||W @ T @ c - W @ y||^2 + ||c / tau||^2?
    # Actually, standard Ridge is (T.T T + lambda I) c = T.T y
    # Weighted Ridge: (T.T T + Lambda) c = T.T y where Lambda = diag(1/tau_k^2)
    
    # Estimate noise level for scaling lambda
    # Use a rough estimate of residual variance from a simple mean fit
    y_mean = np.mean(y)
    residual_var = np.var(y - y_mean)
    
    # Prior variances tau_k^2. 
    # We want high-order terms to be small unless data strongly supports them.
    # Let tau_0 = sqrt(residual_var) * some_factor? 
    # Actually, let's use a fixed decay rate relative to the data scale.
    # The magnitude of c_k depends on the scale of y.
    # Let's scale y to unit variance for the fit, then scale back?
    # No, let's just use a heuristic lambda.
    
    # If we use tau_k = tau_0 * (decay_rate)^k
    # Then penalty term is (c_k / tau_k)^2
    # Let's choose decay_rate = 0.5. So c_5 is penalized 32x more than c_0.
    # And tau_0 should be related to the typical coefficient magnitude.
    
    # Let's solve the unweighted LS first to get a scale reference
    try:
        c_unweighted, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
        c_scale = np.mean(np.abs(c_unweighted))
    except:
        c_scale = np.std(y)
        
    if c_scale < 1e-12:
        c_scale = 1.0

    # Define prior variances
    # We want the regularization to be strong enough to smooth, but not kill the signal.
    # Let's set the penalty weight for k=0 to be small, and increase for higher k.
    # Lambda_k = 1 / (tau_k^2)
    # Let tau_k = c_scale * (0.5)^k * 10.0  (Heuristic: allow 10x scale for k=0, decaying)
    # This might be too strong. Let's use a data-driven approach.
    
    # Alternative: Use the standard error of the coefficients from the LS fit.
    # cov = sigma^2 (T.T T)^-1
    # If |c_k| < 2 * SE_k, it's not significant.
    # We can dampen coefficients based on their significance.
    
    try:
        # Compute covariance matrix of coefficients
        # Assuming homoscedastic noise
        residuals = y - T @ c_unweighted
        sigma2 = np.var(residuals)
        
        # Invert T.T T
        G = T.T @ T
        G_inv = np.linalg.inv(G)
        
        # Standard errors
        se = np.sqrt(sigma2 * np.diag(G_inv))
        
        # Damping strategy:
        # Keep c_k if it is significant (|c_k| > 2*se_k)
        # Otherwise, shrink it towards 0.
        # This is a structural change: Adaptive Model Selection via Damping.
        
        c_cheb = np.zeros(6)
        for k in range(6):
            if se[k] < 1e-15:
                c_cheb[k] = c_unweighted[k]
            else:
                # Soft thresholding / Damping
                # If |c_k| is small relative to noise, set to 0.
                # If large, keep it.
                # Use a factor of 2 for "significance"
                threshold = 2.0 * se[k]
                if abs(c_unweighted[k]) > threshold:
                    # Keep, but maybe shrink slightly? No, keep LS estimate.
                    c_cheb[k] = c_unweighted[k]
                else:
                    # Damp to zero (or small fraction)
                    c_cheb[k] = 0.0
                    
    except np.linalg.LinAlgError:
        # Fallback to unweighted
        c_cheb = c_unweighted

    # Convert Chebyshev to Monomial
    alpha = 1.0 / A
    beta = -B / A
    
    a_monomial = np.zeros(6)
    
    d_coeffs = np.zeros((6, 6))
    d_coeffs[0, 0] = 1.0
    d_coeffs[1, 1] = 1.0
    d_coeffs[2, 2] = 2.0; d_coeffs[2, 0] = -1.0
    d_coeffs[3, 3] = 4.0; d_coeffs[3, 1] = -3.0
    d_coeffs[4, 4] = 8.0; d_coeffs[4, 2] = -8.0; d_coeffs[4, 0] = 1.0
    d_coeffs[5, 5] = 16.0; d_coeffs[5, 3] = -20.0; d_coeffs[5, 1] = 5.0
    
    for k in range(6):
        ck = c_cheb[k]
        if abs(ck) < 1e-15:
            continue
            
        for j in range(k + 1):
            dj = d_coeffs[k, j]
            if abs(dj) < 1e-15:
                continue
                
            for m in range(j + 1):
                binom_jm = 1.0
                for q in range(m):
                    binom_jm *= (j - q) / (q + 1)
                
                term = dj * binom_jm * (alpha ** m) * (beta ** (j - m))
                a_monomial[m] += ck * term

    if not np.all(np.isfinite(a_monomial)):
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            a_monomial = np.zeros(6)
            a_monomial[:2] = c_lin
        except:
            a_monomial = np.zeros(6)

    return a_monomial
# EVOLVE-BLOCK-END
