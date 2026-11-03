"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a sensitivity-based local optimization that targets peak reduction.

Key improvements:
- FFT-based autoconvolution and sensitivity computation (O(N log N) vs O(N^2))
- Reduced parametric search to save time for refinement
- Final polish pass with adaptive step sizes
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill
FFT_SIZE = 2 * N  # for linear convolution via FFT

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F, size)
    return C[:2*n-1]

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
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def compute_sensitivity_fft(h, peak_mask):
    """Compute sensitivity of autoconvolution peaks to each bin using FFT.
    
    For autoconvolution c[m] = sum_j h[j]*h[m-j], the gradient w.r.t. h[j]
    is sum_m peak_mask[m] * h[m-j] (where m-j is valid).
    
    This can be computed as: for each j, sum over m in peaks of h[m-j].
    Equivalently, this is a cross-correlation of h with the peak mask.
    """
    n = len(h)
    # We need: sens[j] = sum_{m: peak_mask[m]=1} h[m-j] for valid m-j in [0,n-1]
    # This is the convolution of peak_mask with h, but we need the "reversed" version.
    # sens[j] = sum_m peak_mask[m] * h[m-j]
    # Let k = m-j, then m = j+k, so sens[j] = sum_k peak_mask[j+k] * h[k]
    # This is the cross-correlation of peak_mask with h.
    
    # Use FFT: cross-correlation of a and b is ifft(fft(a) * conj(fft(b)))
    size = 2 * n
    P = np.fft.rfft(peak_mask.astype(np.float64), size)
    H = np.fft.rfft(h, size)
    # Cross-correlation: sum_k a[j+k] * b[k] = ifft(fft(a) * conj(fft(b)))
    corr = np.fft.irfft(P * np.conj(H), size)
    # corr[j] = sum_k peak_mask[j+k] * h[k] for j in [0, size-1]
    # We want j in [0, n-1]
    return corr[:n]

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and reduce bins
    that contribute more to peaks than to off-peak regions.
    Uses FFT for O(N log N) sensitivity computation.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Compute sensitivity using FFT
        sensitivity = compute_sensitivity_fft(current_h, peak_mask)
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final polish pass with adaptive step sizes and tighter convergence."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    threshold_frac = 0.98  # tighter threshold for polish
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time and no_improve_count < max_no_improve:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        sensitivity = compute_sensitivity_fft(current_h, peak_mask)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Adaptive step: smaller when close to convergence
        effective_alpha = alpha * (1.0 + 0.5 * np.exp(-no_improve_count / 10.0))
        
        h_new = current_h * (1.0 - effective_alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components (reduced for speed)
    for n_c in [2, 3, 4]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Fewer betas and restarts to save time for refinement
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
            
            for restart in range(3):
                if time.time() - start_time > TIME_LIMIT - 20:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
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
    
    # Local refinement: sensitivity-based peak reduction (FFT-accelerated)
    if time.time() - start_time < TIME_LIMIT - 15:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass with tighter convergence
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
