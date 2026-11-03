import numpy as np
import time
from scipy.fft import dct, idct

N = 600
TIME_LIMIT = 145  # Safety margin for 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def spectral_to_time(coeffs):
    """
    Convert DCT-II coefficients to time-domain step function.
    coeffs: array of length N (DCT-II coefficients)
    Returns: array of length N (time-domain values, non-negative, sum=1)
    """
    # Inverse DCT-II
    f = idct(coeffs, type=2, norm='ortho')
    
    # Enforce non-negativity
    f = np.maximum(f, 0.0)
    
    # Normalize to sum 1
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    else:
        # Fallback: uniform
        f = np.ones(N) / N
        
    return f

def time_to_spectral(f):
    """
    Convert time-domain step function to DCT-II coefficients.
    """
    # DCT-II
    coeffs = dct(f, type=2, norm='ortho')
    return coeffs

def exact_coordinate_descent(coeffs, time_limit=10.0, max_iters=5000):
    """
    Optimize DCT coefficients using coordinate descent.
    For each coefficient k, we adjust its value to minimize the max autoconvolution.
    Since the mapping from coeffs to f is linear (before projection), and f is 
    projected to simplex, the objective is non-smooth. 
    However, we can approximate the optimal step for each coefficient by 
    evaluating the objective at a few candidate values.
    
    Actually, a more robust approach for spectral optimization is:
    1. Compute current f and score.
    2. For each freq bin k, compute the sensitivity of the max peak to coeff[k].
    3. Perform a small line search or fixed step update.
    
    Given the complexity of the projection, a simpler but effective strategy is:
    - Use the fact that high frequency coefficients cause oscillations (peaks).
    - We want to suppress high frequencies while maintaining the shape.
    - Let's try a direct optimization loop where we perturb one coefficient at a time.
    """
    
    start_time = time.time()
    best_coeffs = coeffs.copy()
    best_f = spectral_to_time(best_coeffs)
    best_score = compute_score(best_f)
    
    # Precompute DCT matrix? No, N=600, idct is fast enough.
    # But doing N idcts per iteration is O(N^2 log N) or similar.
    # Let's use a smaller step size and fewer iterations.
    
    # Strategy: Cyclic coordinate descent with adaptive step size.
    # We will iterate through coefficients. For each, try increasing and decreasing.
    
    step_size = 0.01
    n_coeffs = len(coeffs)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Cycle through a subset of coefficients or all?
        # N=600. 5000 iters / 600 ~ 8 full cycles.
        # Let's just iterate 0..N-1.
        
        k = it % n_coeffs
        
        # Try a few perturbations for coeff[k]
        # We want to reduce the peak. 
        # High k (high freq) usually increases peak if amplitude is large.
        # Low k (low freq) shapes the envelope.
        
        c0 = best_coeffs[k]
        
        # Candidates: c0, c0 + step, c0 - step, c0 + 2*step, c0 - 2*step
        # To save time, just try +step and -step first.
        
        candidates = [c0, c0 + step_size, c0 - step_size]
        
        best_c = c0
        best_s = best_score
        
        for c_new in candidates:
            # Skip if no change
            if abs(c_new - c0) < 1e-9:
                continue
                
            test_coeffs = best_coeffs.copy()
            test_coeffs[k] = c_new
            f_test = spectral_to_time(test_coeffs)
            s_test = compute_score(f_test)
            
            if s_test < best_s:
                best_s = s_test
                best_c = c_new
        
        # Update if improved
        if best_c != c0:
            best_coeffs[k] = best_c
            best_f = spectral_to_time(best_coeffs)
            best_score = best_s
            # Slowly decrease step size
            step_size *= 0.995
        else:
            # If no improvement, shrink step size for this bin? 
            # Or just continue.
            pass
            
    return best_coeffs, best_f, best_score

def construct():
    start_time = time.time()
    
    # 1. Initial Guess
    # Start with a smooth, low-pass filtered signal.
    # Uniform distribution has flat spectrum (DC only).
    # A Gaussian in time is a Gaussian in frequency.
    # Let's start with a Gaussian in time, then go to spectral domain.
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Initial time domain: Gaussian centered at 0
    f_init = np.exp(-0.5 * (x / 0.05) ** 2)
    f_init = np.maximum(f_init, 0)
    f_init /= np.sum(f_init)
    
    # Convert to spectral
    coeffs = time_to_spectral(f_init)
    
    # 2. Spectral Coordinate Descent
    # Time budget: 120s for optimization, 10s for final checks.
    opt_time_limit = 120.0
    coeffs_opt, f_opt, score_opt = exact_coordinate_descent(coeffs, time_limit=opt_time_limit)
    
    # 3. Final Polish: Direct time-domain coordinate descent on the result
    # Sometimes spectral optimization gets stuck in local minima due to the projection.
    # A quick pass in time domain can help.
    
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5.0
    
    if remaining_time > 5.0:
        f_current = f_opt.copy()
        s_current = score_opt
        
        # Simple coordinate descent on time domain
        # For each bin j, try to adjust f[j] to reduce max peak.
        # This is O(N^2) per pass if done naively, but we can do a few passes.
        
        # Let's do a "peak shaving" pass.
        # Identify max peak location.
        conv = np.convolve(f_current, f_current, mode='full')
        k_max = np.argmax(conv)
        
        # Sensitivity of conv[k_max] to f[j]
        # d conv[k_max] / d f[j] = f[k_max - j] + f[j - k_max]
        
        # We want to reduce f[j] where sensitivity is high, and increase where low.
        # But we must maintain sum=1 and f>=0.
        
        # Let's do a few iterations of this.
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 5.0:
                break
                
            conv = np.convolve(f_current, f_current, mode='full')
            k_max = np.argmax(conv)
            
            sens = np.zeros(N)
            for j in range(N):
                idx1 = k_max - j
                idx2 = j - k_max
                val = 0.0
                if 0 <= idx1 < N:
                    val += f_current[idx1]
                if 0 <= idx2 < N:
                    val += f_current[idx2]
                sens[j] = val
                
            # Normalize sensitivity to [0, 1] roughly
            sens_max = np.max(sens)
            if sens_max > 0:
                sens_norm = sens / sens_max
            else:
                break
                
            # Move mass from high sens to low sens
            # Step size
            step = 0.01
            mask_high = sens_norm > 0.9
            mask_low = sens_norm < 0.1
            
            n_high = np.sum(mask_high)
            n_low = np.sum(mask_low)
            
            if n_high == 0 or n_low == 0:
                break
                
            # Average mass in high and low regions
            mass_high = np.mean(f_current[mask_high])
            mass_low = np.mean(f_current[mask_low])
            
            if mass_high > mass_low:
                # Move some mass
                delta = 0.1 * (mass_high - mass_low)
                
                f_current[mask_high] -= delta
                f_current[mask_low] += delta
                
                f_current = np.maximum(f_current, 0)
                s = np.sum(f_current)
                if s > 1e-15:
                    f_current = f_current / s
                else:
                    break
                    
                new_score = compute_score(f_current)
                if new_score < s_current:
                    s_current = new_score
                else:
                    # Revert
                    # We didn't store previous state, so this is a bit risky.
                    # Let's just break if it doesn't improve to avoid oscillation.
                    break

    # Final check
    if f_opt is None:
        f_opt = f_init
    else:
        # Ensure f_opt is the best one found
        s_opt = compute_score(f_opt)
        s_curr = compute_score(f_current)
        if s_curr < s_opt:
            f_opt = f_current
            
    f_opt = np.maximum(f_opt, 0.0)
    s = np.sum(f_opt)
    if s > 1e-15:
        f_opt = f_opt / s
        
    return [float(x) for x in f_opt]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
