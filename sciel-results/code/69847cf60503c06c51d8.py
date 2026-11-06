import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Subsample for kernel regression if very large
    max_samples = 8000
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        x_s = x[idx]
        t_s = t[idx]
    else:
        x_s = x
        t_s = t
    
    n_s = len(x_s)
    
    # Center and scale x for numerical stability in kernel regression
    mu_x = np.mean(x_s)
    sigma_x = np.std(x_s)
    if sigma_x < 1e-10:
        sigma_x = 1.0
    
    # Use orthogonal Legendre polynomials for the final fit
    # Strategy:
    # 1. Denoise t using kernel regression with various bandwidths
    # 2. Fit degree-5 polynomial in orthogonal basis
    # 3. Select best bandwidth using held-out validation
    # 4. Refit on all data with best bandwidth
    # 5. Convert to standard monomial coefficients
    
    # Precompute: sort x_s for efficient kernel evaluation
    sort_idx = np.argsort(x_s)
    x_sorted = x_s[sort_idx]
    t_sorted = t_s[sort_idx]
    
    # Try multiple bandwidths
    # Silverman's rule as base
    h_base = 1.06 * sigma_x * (n_s ** (-0.2))
    bandwidth_factors = [0.3, 0.5, 0.7, 1.0, 1.3, 1.8, 2.5]
    
    # For validation, use a random split
    rng_val = np.random.RandomState(123)
    perm = rng_val.permutation(n_s)
    split = int(0.75 * n_s)
    fit_idx = perm[:split]
    val_idx = perm[split:]
    
    x_fit = x_s[fit_idx]
    t_fit = t_s[fit_idx]
    x_val = x_s[val_idx]
    t_val = t_s[val_idx]
    
    def _kernel_denoise(x_data, t_data, x_eval, h):
        """Nadaraya-Watson kernel regression using Gaussian kernel."""
        n_eval = len(x_eval)
        result = np.zeros(n_eval)
        
        # Process in chunks to manage memory
        chunk_size = 2000
        for i in range(0, n_eval, chunk_size):
            end = min(i + chunk_size, n_eval)
            x_chunk = x_eval[i:end]
            
            # Compute distances
            diff = x_chunk[:, np.newaxis] - x_data[np.newaxis, :]
            weights = np.exp(-0.5 * (diff / h) ** 2)
            
            w_sum = np.sum(weights, axis=1)
            wt_sum = np.sum(weights * t_data[np.newaxis, :], axis=1)
            
            result[i:end] = wt_sum / np.maximum(w_sum, 1e-12)
        
        return result
    
    def _fit_legendre_and_convert(x_data, t_data, degree=5):
        """Fit polynomial using orthogonal Legendre basis, return monomial coefficients."""
        if len(x_data) < degree + 1:
            # Not enough data, use lower degree
            degree = max(1, len(x_data) - 1)
        
        # Scale x to [-1, 1] for Legendre
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_range = x_max - x_min
        if x_range < 1e-10:
            x_range = 1.0
        
        x_scaled = (x_data - (x_min + x_max) / 2.0) / (x_range / 2.0)
        
        # Build Legendre polynomial design matrix
        # Use numpy's legendre for stable evaluation
        V = np.zeros((len(x_data), degree + 1))
        for k in range(degree + 1):
            # Evaluate Legendre polynomial P_k at x_scaled
            # Use recurrence for stability
            if k == 0:
                V[:, k] = 1.0
            elif k == 1:
                V[:, k] = x_scaled
            else:
                V[:, k] = ((2*k - 1) * x_scaled * V[:, k-1] - (k - 1) * V[:, k-2]) / k
        
        # Solve least squares
        try:
            coeffs_leg, _, _, _ = np.linalg.lstsq(V, t_data, rcond=None)
        except Exception:
            try:
                coeffs_leg = np.linalg.solve(V.T @ V + 1e-8 * np.eye(degree+1), V.T @ t_data)
            except:
                return None
        
        # Convert from Legendre basis on scaled x to monomial basis on original x
        # x_scaled = (x - mid) / half_range
        # So x = mid + half_range * x_scaled
        # P_k(x_scaled) expressed in powers of x_scaled, then substitute
        
        mid = (x_min + x_max) / 2.0
        half_range = x_range / 2.0
        
        # Convert each Legendre polynomial to monomial in x_scaled
        # P_0 = 1
        # P_1 = x
        # P_2 = (3x^2 - 1)/2
        # P_3 = (5x^3 - 3x)/2
        # P_4 = (35x^4 - 30x^2 + 3)/8
        # P_5 = (63x^5 - 70x^3 + 15x)/8
        
        # Build conversion matrix: monomial_coeffs[i] = sum_j M[i,j] * coeffs_leg[j]
        # where M[i,j] is coefficient of x^i in P_j(x_scaled)
        
        # First, get monomial coefficients of each P_k in x_scaled
        leg_to_mono_scaled = np.zeros((degree + 1, degree + 1))
        leg_to_mono_scaled[0, 0] = 1.0  # P_0
        if degree >= 1:
            leg_to_mono_scaled[1, 1] = 1.0  # P_1
        if degree >= 2:
            leg_to_mono_scaled[0, 2] = -0.5
            leg_to_mono_scaled[2, 2] = 1.5
        if degree >= 3:
            leg_to_mono_scaled[1, 3] = -1.5
            leg_to_mono_scaled[3, 3] = 2.5
        if degree >= 4:
            leg_to_mono_scaled[0, 4] = 3.0/8.0
            leg_to_mono_scaled[2, 4] = -30.0/8.0
            leg_to_mono_scaled[4, 4] = 35.0/8.0
        if degree >= 5:
            leg_to_mono_scaled[1, 5] = 15.0/8.0
            leg_to_mono_scaled[3, 5] = -70.0/8.0
            leg_to_mono_scaled[5, 5] = 63.0/8.0
        
        # Now substitute x_scaled = (x - mid) / half_range
        # (x - mid)^i / half_range^i = sum_{j=0}^{i} C(i,j) * x^j * (-mid)^{i-j} / half_range^i
        
        mono_coeffs = np.zeros(degree + 1)
        for j in range(degree + 1):  # Legendre degree
            for i in range(j + 1):  # power in x_scaled
                c = leg_to_mono_scaled[i, j]
                if abs(c) < 1e-15:
                    continue
                # (x - mid)^i / half_range^i
                # = sum_{k=0}^{i} C(i,k) * x^k * (-mid)^{i-k} / half_range^i
                for k in range(i + 1):
                    # C(i,k)
                    binom = 1.0
                    for m in range(k):
                        binom *= (i - m) / (m + 1)
                    term = c * binom * ((-mid) ** (i - k)) / (half_range ** i)
                    mono_coeffs[k] += term * coeffs_leg[j]
        
        # Pad to 6 coefficients
        result = np.zeros(6)
        result[:degree+1] = mono_coeffs[:min(degree+1, 6)]
        return result
    
    best_coeff = None
    best_val_mse = np.inf
    
    for factor in bandwidth_factors:
        if time.time() > deadline - 10.0:
            break
        
        h_val = h_base * factor
        
        # Denoise fit set
        t_denoised_fit = _kernel_denoise(x_fit, t_fit, x_fit, h_val)
        
        # Fit polynomial on denoised fit set
        coeff_candidate = _fit_legendre_and_convert(x_fit, t_denoised_fit, 5)
        if coeff_candidate is None:
            continue
        
        # Evaluate on validation set (also denoised)
        t_denoised_val = _kernel_denoise(x_val, t_val, x_val, h_val)
        pred_val = np.zeros(len(x_val))
        for i in range(6):
            pred_val += coeff_candidate[i] * x_val ** i
        val_mse = np.mean((pred_val - t_denoised_val) ** 2)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            # Store the bandwidth factor for refit
            best_factor = factor
    
    if best_coeff is None and best_val_mse < np.inf:
        # Refit on all data with best bandwidth
        h_best = h_base * best_factor
        t_denoised_all = _kernel_denoise(x_s, t_s, x_s, h_best)
        best_coeff = _fit_legendre_and_convert(x_s, t_denoised_all, 5)
    
    if best_coeff is None:
        # Fallback: direct polynomial fit
        try:
            x_min = np.min(x)
            x_max = np.max(x)
            x_range = x_max - x_min
            if x_range < 1e-10:
                x_range = 1.0
            x_scaled = (x - (x_min + x_max)/2.0) / (x_range/2.0)
            coeffs = np.polynomial.polynomial.polyfit(x_scaled, t, min(5, len(x)-1))
            # Convert back
            mid = (x_min + x_max)/2.0
            hr = x_range/2.0
            result = np.zeros(6)
            for i in range(min(6, len(coeffs))):
                for k in range(i+1):
                    binom = 1.0
                    for m in range(k):
                        binom *= (i-m)/(m+1)
                    result[k] += coeffs[i] * binom * ((-mid)**(i-k)) / (hr**i)
            best_coeff = result
        except:
            best_coeff = np.zeros(6)
    
    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.where(np.isfinite(best_coeff), best_coeff, 0.0)
    
    return best_coeff
