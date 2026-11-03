"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on cosine bumps.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using FFT-based sensitivity and adaptive updates.
4. Add a final polish pass with tighter tolerances.

Key improvements:
- Pure FFT-based sensitivity computation (no loop over peaks)
- Adaptive step size with momentum
- Better time management
- More diverse initialization
- Coordinate descent for final polishing
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
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

def compute_sensitivity_pure_fft(current_h, peak_indices):
    """
    Compute sensitivity using pure FFT.
    
    sensitivity[j] = sum_{m in peaks} current_h[m-j] for valid j
    
    This is equivalent to: sensitivity = ifft(fft(current_h) * conj(fft(peak_ind)))
    where peak_ind is the indicator of peak lags.
    """
    N = len(current_h)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Create peak indicator
    peak_ind = np.zeros(size)
    peak_ind[peak_indices] = 1.0
    
    # FFT of current_h and peak_ind
    H = np.fft.rfft(current_h, n=fft_size)
    P = np.fft.rfft(peak_ind, n=fft_size)
    
    # Cross-correlation: corr[j] = sum_k current_h[k] * peak_ind[j+k]
    # In frequency domain: corr = ifft(conj(H) * P)
    corr = np.fft.irfft(np.conj(H) * P, n=fft_size)
    
    # Extract the valid part: sensitivity[j] for j in [0, N-1]
    # The correlation result has length fft_size, and we need indices 0 to N-1
    sensitivity = corr[:N]
    
    return sensitivity

def coordinate_descent_polish(f, start_time, max_time):
    """
    Coordinate descent polishing: try small perturbations to individual bins.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    s = np.sum(current_f)
    if s < 1e-12:
        return best_f, best_val
    current_f = current_f * (N / s)
    
    # Precompute FFT for fast evaluation
    conv = fft_autocorr(current_f)
    current_val = np.max(conv) / (N * N)
    
    # Try perturbing each bin with small changes
    step_sizes = [0.01, 0.005, 0.001]
    
    for step in step_sizes:
        if time.time() - start_time > max_time:
            break
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for i in range(0, N, 3):  # Stride 3 to save time
                if time.time() - start_time > max_time:
                    break
                
                for delta in [step, -step]:
                    f_trial = current_f.copy()
                    f_trial[i] = max(0.0, current_f[i] + delta)
                    
                    s_trial = np.sum(f_trial)
                    if s_trial < 1e-12:
                        continue
                    f_trial = f_trial * (N / s_trial)
                    
                    conv_trial = fft_autocorr(f_trial)
                    val_trial = np.max(conv_trial) / (N * N)
                    
                    if val_trial < current_val - 1e-12:
                        current_f = f_trial
                        current_val = val_trial
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_f = current_f.copy()
    
    return best_f, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.05, threshold_frac_init=0.85):
    """Local refinement using FFT-based sensitivity and adaptive updates."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    momentum = np.zeros_like(current_h)
    beta_momentum = 0.9
    
    iter_count = 0
    total_iters_estimate = 300
    
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
        
        # Adaptive threshold: start wider, tighten over time
        progress = min(iter_count / total_iters_estimate, 1.0)
        threshold_frac_adaptive = threshold_frac_init + (0.995 - threshold_frac_init) * progress
        threshold = max_c * threshold_frac_adaptive
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        # Compute sensitivity using pure FFT
        sensitivity = compute_sensitivity_pure_fft(current_h, peak_indices)
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Update with momentum
        momentum = beta_momentum * momentum + (1 - beta_momentum) * sens_norm
        h_new = current_h * (1.0 - alpha * momentum)
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
            alpha *= 0.90
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
    momentum = np.zeros_like(current_h)
    beta_momentum = 0.8
    
    iter_count = 0
    total_iters_estimate = 150
    
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
        threshold_frac = 0.995 + 0.0049 * progress
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        sensitivity = compute_sensitivity_pure_fft(current_h, peak_indices)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        momentum = beta_momentum * momentum + (1 - beta_momentum) * sens_norm
        h_new = current_h * (1.0 - alpha * momentum)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.01, 0.1)
        else:
            alpha *= 0.95
            if alpha < 1e-10:
                break
        
        iter_count += 1
    
    return best_h, best_val

def spectral_refinement(f, start_time, max_time):
    """Refine using frequency domain insights."""
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    for iteration in range(500):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autocorr(current_f)
        current_val = np.max(conv) / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Get FFT
        F = np.fft.rfft(current_f, n=2*N)
        power = np.abs(F) ** 2
        
        # Identify frequency bins with high power
        if len(power) > 1:
            power_no_dc = power[1:]
            max_power = np.max(power_no_dc)
            if max_power > 1e-12:
                threshold = max_power * 0.9
                high_power_mask = power_no_dc >= threshold
                F_damped = F.copy()
                F_damped[1:][high_power_mask] *= 0.95
                f_new = np.fft.irfft(F_damped, n=2*N)[:N]
                f_new = np.maximum(f_new, 0.0)
                s_new = np.sum(f_new)
                if s_new > 1e-12:
                    f_new = f_new * (N / s_new)
                    conv_new = fft_autocorr(f_new)
                    val_new = np.max(conv_new) / (N * N)
                    if val_new < current_val:
                        current_f = f_new
                    else:
                        break
                else:
                    break
        else:
            break
    
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 45:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 40:
                    break
                
                # Diverse initialization
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
                    centers = rng.uniform(-0.1, 0.1, n_c)
                    widths = rng.uniform(0.03, 0.15, size=n_c)
                    heights = rng.uniform(1.0, 3.0, size=n_c)
                elif restart == 3:
                    # Wide bumps
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.15, 0.4, size=n_c)
                    heights = rng.uniform(0.3, 1.5, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
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
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral refinement
    if time.time() - start_time < TIME_LIMIT - 15:
        spec_f, spec_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if spec_score < best_score:
            best_f = spec_f
            best_score = spec_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 10
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Coordinate descent for final refinement
    if time.time() - start_time < TIME_LIMIT - 3:
        cd_f, cd_score = coordinate_descent_polish(
            best_f, start_time, TIME_LIMIT - 1
        )
        if cd_score < best_score:
            best_f = cd_f
            best_score = cd_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
