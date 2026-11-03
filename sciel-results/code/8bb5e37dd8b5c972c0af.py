"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a parametric family based on cosine bumps for good initial solutions.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction.
4. Spectral-domain refinement with adaptive learning rate.
5. Multi-scale polish pass with varying thresholds.

Key improvements:
- Better time allocation across stages
- More aggressive spectral refinement
- Multi-scale polish for finer convergence
- FFT-based autoconvolution for speed
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
    """Compute sensitivity using FFT cross-correlation."""
    N = len(current_h)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Create peak indicator
    peak_ind = np.zeros(fft_size)
    peak_ind[peak_indices] = 1.0
    
    # Zero pad f
    f_padded = np.zeros(fft_size)
    f_padded[:N] = current_h
    
    # Cross-correlation: s[j] = sum_k f[k] I_P[j+k]
    # In frequency domain: F * conj(I_P)
    F = np.fft.rfft(f_padded)
    IP = np.fft.rfft(peak_ind)
    
    S = np.fft.irfft(F * np.conj(IP), n=fft_size)
    
    # The linear correlation is in the first N elements
    return S[:N]

def spectral_refinement(h, start_time, max_time):
    """Spectral-domain refinement using projected gradient descent with adaptive learning rate."""
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
    lr = 0.03
    min_lr = 1e-7
    no_improve_count = 0
    max_no_improve = 50
    
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
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        # Find peak lags with tighter threshold for better focus
        threshold = max_c * 0.97
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        # Compute sensitivity in time domain
        sensitivity = compute_sensitivity_fft(h_recon, peak_indices)
        
        # Transform sensitivity to frequency domain
        S = np.fft.rfft(sensitivity, n=fft_size)
        
        # Gradient step in frequency domain with normalization correction
        # The normalized objective is max(conv)/(sum(f))^2
        # We approximate the gradient direction using the sensitivity
        H_new = H - lr * S
        
        # Reconstruct and project to nonnegativity
        h_new = np.fft.irfft(H_new, n=fft_size)[:N]
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            lr *= 0.5
            if lr < min_lr:
                break
            continue
        
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val * (1.0 - 1e-10):  # Require strict improvement
            H = np.fft.rfft(h_new, n=fft_size)
            current_h = h_new
            lr = min(lr * 1.015, 0.25)
            no_improve_count = 0
        else:
            lr *= 0.92
            if lr < min_lr:
                break
    
    return best_h, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.02, threshold_frac=0.98):
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
        threshold = max_c * threshold_frac
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
            alpha = min(alpha * 1.02, 0.2) # Conservative growth
        else:
            alpha *= 0.95 # Slower decay to allow more tries
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Multi-scale polish pass with varying thresholds and adaptive step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    # Multiple scales: start with moderate steps, tighten over time
    scales = [
        {'alpha': 0.01, 'threshold': 0.99, 'name': 'coarse'},
        {'alpha': 0.005, 'threshold': 0.995, 'name': 'medium'},
        {'alpha': 0.002, 'threshold': 0.999, 'name': 'fine'},
        {'alpha': 0.001, 'threshold': 0.9995, 'name': 'ultra_fine'},
    ]
    
    # Allocate time proportionally to scales
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining < 1.0:
        return best_h, best_val
    
    # Give more time to finer scales
    time_weights = [0.15, 0.25, 0.35, 0.25]
    
    for scale_idx, scale in enumerate(scales):
        scale_time = time_weights[scale_idx] * remaining
        scale_deadline = time.time() + scale_time
        
        alpha = scale['alpha']
        threshold_frac = scale['threshold']
        min_alpha = 1e-6
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < scale_deadline and time.time() - start_time < max_time:
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
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            
            # Tight threshold for polish
            threshold = max_c * threshold_frac
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
            
            if val_new < current_val * (1.0 - 1e-12):
                current_h = h_new
                alpha = min(alpha * 1.01, 0.05)
                no_improve = 0
            else:
                alpha *= 0.97
                if alpha < min_alpha:
                    break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try box function initialization
    # A box function is a strong candidate for low autocorrelation sidelobes
    box_widths = [0.1, 0.2, 0.3, 0.4]
    for w in box_widths:
        f_box = np.zeros(N)
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        mask = np.abs(x) <= w / 2.0
        f_box[mask] = 1.0
        score_box = compute_score(f_box)
        if score_box < best_score:
            best_f = f_box
            best_score = score_box

    # Try different numbers of cosine bump components with better initialization
    # Reduced time budget for parametric optimization to allow more local refinement
    for n_c in [3, 4]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 30:
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
    # Allocate ~20% of remaining time
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 10.0:
        local_time = min(remaining * 0.20, 30.0)
        local_deadline = time.time() + local_time
        refined_f, refined_score = local_refinement(
            best_f, start_time, local_deadline
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral refinement: work in frequency domain
    # Allocate ~35% of remaining time
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 15.0:
        spectral_time = min(remaining * 0.35, 50.0)
        spectral_deadline = time.time() + spectral_time
        spectral_f, spectral_score = spectral_refinement(
            best_f, start_time, spectral_deadline
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    # Final polish pass: use all remaining time
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 5.0:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # One more local refinement if time permits
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 10.0:
        final_local_time = min(remaining * 0.30, 25.0)
        final_local_deadline = time.time() + final_local_time
        final_f, final_score = local_refinement(
            best_f, start_time, final_local_deadline
        )
        if final_score < best_score:
            best_f = final_f
            best_score = final_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
