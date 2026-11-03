"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a sensitivity-based local optimization that targets peak reduction.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. We use a smooth max (log-sum-exp) surrogate for the parametric
optimization, then apply a gradient-like local refinement that reduces the
sensitivity of peak lags to reduce the maximum autoconvolution value.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

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

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction with
    adaptive threshold and coordinate descent polish.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    n_fft = 2 * N  # size for full convolution via FFT
    stall_count = 0
    max_stall = 30
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Use FFT for fast convolution
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Try coordinate descent polish
                current_h = _coordinate_descent_polish(current_h, start_time, max_time)
                s = np.sum(current_h)
                if s < 1e-12:
                    break
                current_h = current_h * (N / s)
                f_hat = np.fft.rfft(current_h, n=n_fft)
                conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
                current_val = np.max(conv) / (N * N)
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                    stall_count = 0
                else:
                    break
        
        # Adaptive threshold: tighten as we converge
        threshold_frac = max(0.90, 0.95 - 0.01 * stall_count)
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Compute sensitivity via FFT
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Evaluate new candidate
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def _coordinate_descent_polish(h, start_time, max_time):
    """Coordinate descent: for each bin, find optimal value that minimizes
    the peak autoconvolution contribution.
    """
    n_fft = 2 * N
    h = h.copy()
    s = np.sum(h)
    if s < 1e-12:
        return h
    h = h * (N / s)
    
    # Precompute base convolution without each bin's contribution
    # For bin i, conv[i+j] includes h[i]*h[j] for all j
    # We'll do a few passes of coordinate updates
    
    for _pass in range(3):
        if time.time() - start_time > max_time - 5:
            break
        
        for i in range(N):
            if time.time() - start_time > max_time - 5:
                break
            
            old_val = h[i]
            if old_val < 1e-15:
                continue
            
            # Try reducing this bin: set to 0 and see effect
            # Quick estimate: contribution to peak from bin i
            # conv contains terms h[i]*h[j] for all j at lag i-j and j-i
            
            # Binary search for optimal h[i]
            lo, hi = 0.0, old_val * 2.0
            for _ in range(10):
                mid = (lo + hi) / 2.0
                h[i] = mid
                s = np.sum(h)
                if s < 1e-12:
                    h[i] = old_val
                    break
                h_temp = h * (N / s)
                f_hat = np.fft.rfft(h_temp, n=n_fft)
                conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
                val = np.max(conv) / (N * N)
                
                # Compare with keeping old value
                h[i] = old_val
                s_old = np.sum(h)
                h_old = h * (N / s_old)
                f_hat_old = np.fft.rfft(h_old, n=n_fft)
                conv_old = np.fft.irfft(f_hat_old * f_hat_old, n=n_fft)[:2*N-1]
                val_old = np.max(conv_old) / (N * N)
                
                if val < val_old:
                    lo = mid
                    h[i] = mid
                else:
                    hi = mid
                    h[i] = old_val
            
            # Restore best value found
            h[i] = old_val
            s = np.sum(h)
            if s > 1e-12:
                h = h * (N / s)
    
    return h

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components with tighter time budget
    # Reserve at least 60% of time for local refinement
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.35
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Fewer restarts, tighter tolerances to save time
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() > param_budget_end:
                break
            
            for restart in range(3):
                if time.time() > param_budget_end:
                    break
                
                centers = rng.uniform(-0.15, 0.15, size=n_c)
                widths = rng.uniform(0.08, 0.25, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-7}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: FFT-based sensitivity peak reduction
    # This is now fast enough to run for most of the remaining time
    if time.time() < TIME_LIMIT - 5:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final high-precision polish: tighter threshold, smaller steps
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.01
        threshold_final = 0.99
        n_fft = 2 * N
        for _ in range(100):
            if time.time() > TIME_LIMIT - 2:
                break
            s = np.sum(best_f)
            if s < 1e-12:
                break
            best_f = best_f * (N / s)
            f_hat = np.fft.rfft(best_f, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            peak_mask = (conv >= max_c * threshold_final).astype(np.float64)
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            h_new = best_f * (1.0 - alpha_small * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                best_f = h_new
                alpha_small = min(alpha_small * 1.02, 0.1)
            else:
                alpha_small *= 0.9
                if alpha_small < 1e-6:
                    break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
