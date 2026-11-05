"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    using an ensemble of Chebyshev polynomial fits on bootstrap resamples.
    
    Strategy:
    1. Map x to Chebyshev domain [-1, 1].
    2. Generate multiple bootstrap resamples of the training data.
    3. For each resample, fit a degree-5 Chebyshev polynomial using least squares.
    4. Convert each Chebyshev fit to monomial basis.
    5. Return the mean of the monomial coefficients across all ensemble members.
    
    This ensemble approach reduces overfitting to specific noise patterns in the 
    training data and provides a more robust estimate of the underlying 
    subgrid tendency, particularly for generalization to unseen regimes.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

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
    
    N = len(x)
    
    # Pre-compute Chebyshev design matrix for the full dataset
    # This is reused for each bootstrap sample by indexing
    t_all = (x - B) / A
    t_all = np.clip(t_all, -1.0, 1.0)
    
    T_all = np.zeros((N, 6))
    T_all[:, 0] = 1.0
    if N > 0:
        T_all[:, 1] = t_all
        for k in range(2, 6):
            T_all[:, k] = 2.0 * t_all * T_all[:, k-1] - T_all[:, k-2]
    
    # Determine number of bootstrap samples
    # We want enough samples to average out noise, but not so many we run out of time.
    # Each fit is O(N*6^2) roughly. 
    # Let's aim for 50-100 samples if fast, fewer if slow.
    # Estimate time for one fit.
    try:
        # Quick test fit
        t0 = time.time()
        _ = np.linalg.lstsq(T_all, y, rcond=None)
        t_fit = time.time() - t0
    except:
        t_fit = 0.01 # Assume 10ms per fit if error
        
    # Budget: 100 seconds for fitting
    # If one fit takes 1ms, we can do 100,000 fits. Cap at 200 for diminishing returns.
    # If one fit takes 100ms, we can do 1000 fits.
    max_samples = 200
    if t_fit > 1e-5:
        estimated_samples = int(100.0 / t_fit)
        n_bootstrap = max(10, min(max_samples, estimated_samples))
    else:
        n_bootstrap = max_samples
        
    # Ensure we don't exceed deadline
    # Add overhead for conversion and averaging
    overhead_per_sample = 0.001 # 1ms estimate for conversion
    total_time_est = n_bootstrap * (t_fit + overhead_per_sample)
    if total_time_est > 100.0:
        n_bootstrap = max(10, int(100.0 / (t_fit + overhead_per_sample)))

    # Fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Store monomial coefficients for each bootstrap sample
    # We will accumulate the sum to avoid storing all matrices
    sum_monomial = np.zeros(6)
    count = 0
    
    # Pre-compute conversion logic parameters
    alpha = 1.0 / A
    beta = -B / A
    
    # Chebyshev to Monomial conversion matrix (6x6)
    # T_k(t) in terms of (alpha*t + beta) ... wait, t = (x-B)/A = alpha*x + beta
    # So x = (t - beta) / alpha
    # We need to express T_k(t) as polynomial in x.
    # Or simply: fit in t, get c_cheb.
    # Then convert c_cheb to c_monomial in x.
    
    # Let's use the explicit conversion matrix derived from Chebyshev polynomials.
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    # t = alpha * x + beta
    # We need to expand (alpha*x + beta)^k.
    
    # Let's build the conversion matrix M such that c_monomial = M @ c_cheb
    # c_cheb[k] is coeff of T_k(t)
    # T_k(t) = sum_{j=0}^k D[k,j] * t^j
    # Then t^j = (alpha*x + beta)^j = sum_{m=0}^j Binom(j,m) * alpha^m * beta^(j-m) * x^m
    # So c_monomial[m] = sum_{k=m}^5 c_cheb[k] * sum_{j=m}^k D[k,j] * Binom(j,m) * alpha^m * beta^(j-m)
    
    # Let's precompute this matrix M (6x6)
    M = np.zeros((6, 6)) # M[m, k] is contribution of c_cheb[k] to c_monomial[m]
    
    # D[k, j] coeff of t^j in T_k(t)
    D = np.zeros((6, 6))
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    D[2, 2] = 2.0; D[2, 0] = -1.0
    D[3, 3] = 4.0; D[3, 1] = -3.0
    D[4, 4] = 8.0; D[4, 2] = -8.0; D[4, 0] = 1.0
    D[5, 5] = 16.0; D[5, 3] = -20.0; D[5, 1] = 5.0
    
    # Binomial coefficients
    def binom(n, k):
        if k < 0 or k > n:
            return 0
        res = 1
        for i in range(k):
            res *= (n - i) / (i + 1)
        return res
    
    for k in range(6):
        for j in range(k + 1):
            if D[k, j] == 0:
                continue
            for m in range(j + 1):
                term = D[k, j] * binom(j, m) * (alpha ** m) * (beta ** (j - m))
                M[m, k] += term
    
    # Now loop over bootstrap samples
    for i in range(n_bootstrap):
        if time.time() > DEADLINE:
            break
            
        # Bootstrap resample indices
        # Using 'with replacement'
        indices = rng.integers(0, N, size=N)
        
        T_sample = T_all[indices, :]
        y_sample = y[indices]
        
        try:
            # Solve least squares
            c_cheb, _, _, _ = np.linalg.lstsq(T_sample, y_sample, rcond=None)
            
            if not np.all(np.isfinite(c_cheb)):
                continue
                
            # Convert to monomial
            c_monomial = M @ c_cheb
            
            if np.all(np.isfinite(c_monomial)):
                sum_monomial += c_monomial
                count += 1
                
        except Exception:
            continue
            
    if count > 0:
        result = sum_monomial / count
    else:
        # Fallback to single fit on full data
        try:
            c_cheb, _, _, _ = np.linalg.lstsq(T_all, y, rcond=None)
            result = M @ c_cheb
        except:
            result = np.zeros(6)
            
    if not np.all(np.isfinite(result)):
        result = np.zeros(6)
        
    return result
# EVOLVE-BLOCK-END
