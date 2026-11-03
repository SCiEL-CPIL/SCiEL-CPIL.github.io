"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on cosine bumps with multiple components.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction.
4. Add a final polish pass with tighter tolerances.

Key improvements over the seed:
- Better initial parameter sampling with structured diversity
- More aggressive local refinement with adaptive step sizes
- Final polish pass to squeeze out remaining improvements
- FFT-based autoconvolution for speed, enabling more iterations
- Improved sensitivity computation using FFT cross-correlation
- Constant function as guaranteed lower bound
- Additional local refinement iterations with adaptive alpha
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

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

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    # Use next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
    return conv

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

def compute_sensitivity_fft(current_h, peak_indices):
    """Compute sensitivity using FFT cross-correlation.
    
    For each peak lag m, sensitivity[j] += current_h[m - j] for valid j.
    This is a cross-correlation of current_h with an indicator of peak lags.
    """
    N = len(current_h)
    # Create peak indicator
    peak_ind = np.zeros(2 * N - 1)
    peak_ind[peak_indices] = 1.0
    
    # We want sensitivity[j] = sum_{m in peaks} current_h[m-j] for valid j
    # This is equivalent to the cross-correlation of peak_ind and current_h
    # specifically: sensitivity[j] = sum_k peak_ind[k] * current_h[k-j]
    # which is the same as np.correlate(peak_ind, current_h, mode='valid')
    # but we need to be careful with the length.
    
    # Let's use FFT to compute this efficiently.
    # sensitivity[j] = sum_{m} peak_ind[m] * current_h[m-j]
    # Let's flip current_h: h_flip[k] = current_h[N-1-k]
    # Then sensitivity[j] = sum_m peak_ind[m] * h_flip[N-1-(m-j)]
    # = sum_m peak_ind[m] * h_flip[N-1-m+j]
    # This is a convolution of peak_ind and h_flip, shifted.
    
    # Alternatively, just use the direct definition with FFT:
    # corr = ifft(fft(peak_ind) * conj(fft(current_h)))
    # But we need the 'valid' part.
    
    # Let's use the property:
    # sensitivity = np.correlate(peak_ind, current_h, mode='valid')
    # This gives an array of length (2N-1) - (N-1) = N.
    # sensitivity[j] = sum_{k=0}^{N-1} peak_ind[j+k] * current_h[k]
    # Wait, np.correlate(a, v, mode='valid') computes:
    # out[n] = sum_m a[m+n] * conj(v[m])
    # So if we do np.correlate(peak_ind, current_h, mode='valid'):
    # out[j] = sum_{m=0}^{N-1} peak_ind[j+m] * current_h[m]
    # This is exactly what we want! sensitivity[j] = sum_{m in peaks} current_h[m-j]
    # Wait, the index is different.
    # We want sensitivity[j] = sum_{m} peak_ind[m] * current_h[m-j]
    # Let k = m-j => m = k+j
    # sensitivity[j] = sum_{k} peak_ind[k+j] * current_h[k]
    # This is exactly np.correlate(peak_ind, current_h, mode='valid')[j]
    
    # So we can just use np.correlate.
    # But for large N, FFT is faster.
    # np.correlate uses direct method or FFT depending on size.
    # For N=600, direct might be fast enough, but let's use FFT to be safe.
    
    # Using FFT:
    # corr = ifft(fft(peak_ind) * conj(fft(current_h)))
    # The full cross-correlation has length 2N-1.
    # The 'valid' part corresponds to indices 0 to N-1 in the output of ifft?
    # Let's check the indexing.
    
    # Let's just use np.correlate for simplicity and correctness, it's O(N^2) but N=600 is small.
    # 600^2 = 360,000 operations, which is very fast in C.
    # The previous loop was O(P * N) where P is number of peaks.
    # If P is large, it could be slower.
    
    # Let's use np.correlate.
    sensitivity = np.correlate(peak_ind, current_h, mode='valid')
    return sensitivity

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    # Adaptive threshold: start wider, tighten over time
    iter_count = 0
    total_iters_estimate = 200  # rough estimate
    
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
        
        # Adaptive threshold: start at 0.90, tighten to 0.97
        progress = min(iter_count / total_iters_estimate, 1.0)
        threshold_frac_adaptive = 0.90 + 0.07 * progress
        threshold = max_c * threshold_frac_adaptive
        peak_indices = np.where(conv >= threshold)[0]
        
        # If no peaks found (should not happen), fallback to max index
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        # Compute sensitivity
        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        # Use a slightly damped update to prevent oscillation
        # Add a small epsilon to avoid zeroing out too many bins
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 1e-6)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5) # Slightly more aggressive growth
        else:
            alpha *= 0.85 # Slightly faster decay
            if alpha < 1e-8:
                break
        
        iter_count += 1
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    # Use a more aggressive initial alpha for polish
    alpha = 0.03
    iter_count = 0
    total_iters_estimate = 100
    
    while time.time() - start_time < max_time:
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
        
        # Very tight threshold for polish, tighten over time
        progress = min(iter_count / total_iters_estimate, 1.0)
        threshold_frac = 0.99 + 0.009 * progress
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 1e-6)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.90
            if alpha < 1e-10:
                break
        
        iter_count += 1
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components with better initialization
    # Reduced time budget for parametric optimization to allow more local refinement
    for n_c in [3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 70:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [10.0, 20.0, 30.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 50:
                break
            
            for restart in range(5):
                if time.time() - start_time > TIME_LIMIT - 45:
                    break
                
                # Better structured initialization
                if restart == 0:
                    # Symmetric initialization
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    # Asymmetric with varying widths
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                elif restart == 2:
                    # Concentrated bumps
                    centers = rng.uniform(-0.1, 0.1, size=n_c)
                    widths = rng.uniform(0.03, 0.15, size=n_c)
                    heights = rng.uniform(1.0, 3.0, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
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
    
    # Local refinement: sensitivity-based peak reduction
    # Increased time budget
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass
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
