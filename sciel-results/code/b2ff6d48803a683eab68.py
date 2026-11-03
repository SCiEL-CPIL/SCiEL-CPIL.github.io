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
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

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
    
    # sensitivity[j] = sum over peak m of current_h[m-j] where 0<=m-j<N and 0<=m<2N-1
    # This is equivalent to: sensitivity = cross-correlation of current_h with peak_ind
    # Using FFT: sensitivity[j] = sum_m peak_ind[m] * current_h[m-j]
    # = (current_h * peak_ind_flipped)[j] where * is cross-correlation
    # In convolution terms: sensitivity[j] = sum_k current_h[k] * peak_ind[j+k]
    # Let's use np.convolve with mode='full' on current_h and reversed peak_ind
    
    # Actually, sensitivity[j] = sum_{m in peaks} current_h[m-j] for valid indices
    # Let's just do it directly but vectorized where possible
    sensitivity = np.zeros(N)
    for m in peak_indices:
        # j ranges from max(0, m-(N-1)) to min(N-1, m)
        j_min = max(0, m - (N - 1))
        j_max = min(N - 1, m)
        if j_max >= j_min:
            # sensitivity[j] += current_h[m - j] for j in [j_min, j_max]
            # m - j ranges from m - j_max to m - j_min
            idx = m - np.arange(j_min, j_max + 1)
            sensitivity[j_min:j_max+1] += current_h[idx]
    return sensitivity

def spectral_refinement(h, start_time, max_time):
    """Spectral-domain refinement using projected gradient descent.
    
    Work in frequency domain: minimize max of autoconvolution subject to
    nonnegativity in time domain. Use soft-thresholding for nonnegativity.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    # Normalize
    s = np.sum(current_h)
    if s < 1e-12:
        return best_h, best_val
    current_h = current_h * (N / s)
    
    # Compute FFT
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    H = np.fft.rfft(current_h, n=fft_size)
    
    # Learning rate schedule
    lr = 0.05
    min_lr = 1e-6
    
    while time.time() - start_time < max_time:
        # Reconstruct time domain from frequency
        h_recon = np.fft.irfft(H, n=fft_size)[:N]
        h_recon = np.maximum(h_recon, 0.0)
        
        # Normalize
        s_recon = np.sum(h_recon)
        if s_recon < 1e-12:
            break
        h_recon = h_recon * (N / s_recon)
        
        # Compute score
        conv = fft_autocorr(h_recon)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h_recon.copy()
        
        # Find peak lags
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        # Compute gradient in frequency domain
        # The objective is max(conv) / (sum(f))^2
        # Gradient of conv at peak lag m w.r.t. f[j] is f[m-j]
        # In frequency domain, this becomes multiplication by appropriate kernel
        
        # Compute sensitivity in time domain
        sensitivity = compute_sensitivity_fft(h_recon, peak_indices)
        
        # Transform sensitivity to frequency domain
        S = np.fft.rfft(sensitivity, n=fft_size)
        
        # Gradient step in frequency domain
        # We want to reduce peaks, so we subtract lr * sensitivity
        # In frequency domain: H_new = H - lr * S / (sum(f)^2)
        # But we need to be careful with the normalization
        
        # Compute gradient of the normalized objective
        # d/dH [max(conv) / (sum(f))^2] 
        # The max(conv) part gives us the sensitivity kernel
        # The 1/(sum(f))^2 part gives us a correction term
        
        # Simplified: just use the sensitivity as gradient direction
        H_new = H - lr * S
        
        # Enforce conjugate symmetry (already handled by rfft/irfft)
        
        # Reconstruct and project to nonnegativity
        h_new = np.fft.irfft(H_new, n=fft_size)[:N]
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            # Reset and try smaller step
            lr *= 0.5
            if lr < min_lr:
                break
            continue
        
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            H = np.fft.rfft(h_new, n=fft_size)
            current_h = h_new
            lr = min(lr * 1.02, 0.3)
        else:
            lr *= 0.9
            if lr < min_lr:
                break
    
    return best_h, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
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
        # Use a tighter threshold to focus on dominant peaks
        threshold = max_c * 0.95
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
            alpha = min(alpha * 1.05, 0.5) # Slightly more aggressive growth
        else:
            alpha *= 0.85 # Slightly faster decay
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
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
        
        # Very tight threshold for polish
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
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
            alpha = min(alpha * 1.03, 0.2)
        else:
            alpha *= 0.88
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components with better initialization
    # Reduced time budget for parametric optimization to allow more local refinement
    for n_c in [3, 4]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 40:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 35:
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
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
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
    if time.time() - start_time < TIME_LIMIT - 30:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 25
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral refinement: work in frequency domain
    if time.time() - start_time < TIME_LIMIT - 10:
        spectral_f, spectral_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 2:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT
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
