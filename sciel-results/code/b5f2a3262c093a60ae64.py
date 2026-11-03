"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as |sum of Gaussian modes|^2, which is
inherently nonnegative and smooth. Optimize real/imaginary coefficients of
Gaussian modes via Nelder-Mead to minimize the peak of the autoconvolution.
Then apply FFT-based local refinement and a high-precision polish pass.
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

def gaussian_modes(x, centers, sigmas, re_coeffs, im_coeffs):
    """Compute complex signal from Gaussian modes: sum_k (a_k + i*b_k) * exp(-0.5*((x-c_k)/sigma_k)^2)"""
    n_modes = len(centers)
    signal = np.zeros(len(x), dtype=np.complex128)
    for k in range(n_modes):
        g = np.exp(-0.5 * ((x - centers[k]) / sigmas[k]) ** 2)
        signal += (re_coeffs[k] + 1j * im_coeffs[k]) * g
    return signal

def parametric_to_step(params, n_modes):
    """Convert parameters to a step function of length N.
    
    params layout: [centers(n_modes), sigmas(n_modes), re_coeffs(n_modes), im_coeffs(n_modes)]
    Total: 4 * n_modes parameters
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    centers = params[:n_modes]
    sigmas = np.abs(params[n_modes:2*n_modes]) + 0.005
    sigmas = np.minimum(sigmas, 0.25)
    re_coeffs = params[2*n_modes:3*n_modes]
    im_coeffs = params[3*n_modes:4*n_modes]
    
    signal = gaussian_modes(x, centers, sigmas, re_coeffs, im_coeffs)
    f = np.abs(signal) ** 2
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_modes, beta=10.0):
    f = parametric_to_step(params, n_modes)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def fast_peak_shaving(h, start_time, max_time):
    """Fast local refinement using FFT-based convolution and gradient-like steps.
    
    Targets the maximum of the autoconvolution by reducing bins that contribute
    most to the peak, while maintaining non-negativity and constant sum.
    Uses multi-peak sensitivity: considers all lags near the maximum.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.01
    patience = 0
    max_patience = 800
    step_count = 0
    n_peaks = 5  # consider top n_peaks lags
    
    # Precompute FFT size for efficiency
    fft_size = 2 * N
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Compute autoconvolution using FFT
        f_fft = np.fft.rfft(current_h, n=fft_size)
        conv_fft = f_fft * f_fft
        conv = np.fft.irfft(conv_fft, n=fft_size)
        conv = conv[:2*N-1]
        
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find indices of top peaks
        peak_indices = np.argsort(conv)[-n_peaks:]
        
        # Compute combined gradient from all peak lags
        g = np.zeros(N)
        for k_star in peak_indices:
            j_min = max(0, k_star - (N - 1))
            j_max = min(N - 1, k_star)
            if j_min <= j_max:
                indices = np.arange(j_min, j_max + 1)
                f_indices = k_star - indices
                g[indices] += current_h[f_indices]
        
        # Center g to preserve sum
        g_mean = np.mean(g)
        g = g - g_mean
        
        # Adaptive step size
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        # Scale step to be relative to function magnitude
        h_scale = np.mean(current_h)
        alpha_effective = alpha * h_scale / g_norm
        
        # Update
        h_new = current_h - alpha_effective * g
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Check improvement
        val_new = compute_score(h_new)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.5)
            step_count += 1
        else:
            alpha *= 0.95
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """High-precision polish: very small steps with tighter tolerance.
    
    Uses coordinate-wise perturbations for final refinement.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    fft_size = 2 * N
    alpha = 0.001
    no_improve = 0
    max_no_improve = 200
    
    # Precompute indices for coordinate perturbations
    n_coords = N
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        current_h = current_h * (N / s)
        
        # Compute autoconvolution
        f_fft = np.fft.rfft(current_h, n=fft_size)
        conv_fft = f_fft * f_fft
        conv = np.fft.irfft(conv_fft, n=fft_size)
        conv = conv[:2*N-1]
        
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val - 1e-15:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Find peak
        k_star = np.argmax(conv)
        
        # Try small perturbations on a subset of coordinates
        # Use the sensitivity vector to guide which coordinates to perturb
        g = np.zeros(N)
        j_min = max(0, k_star - (N - 1))
        j_max = min(N - 1, k_star)
        if j_min <= j_max:
            indices = np.arange(j_min, j_max + 1)
            f_indices = k_star - indices
            g[indices] = current_h[f_indices]
        
        # Perturb coordinates with highest sensitivity first
        # Try moving mass from high-sensitivity to low-sensitivity regions
        sorted_idx = np.argsort(g)[::-1]
        
        improved = False
        for trial in range(min(20, n_coords)):
            if time.time() - start_time > max_time:
                break
            
            # Pick a high-sensitivity index and a low-sensitivity index
            i_high = sorted_idx[trial % n_coords]
            i_low = (i_high + n_coords // 2) % n_coords
            
            if current_h[i_high] < 1e-10:
                continue
            
            delta = alpha * current_h[i_high]
            if delta < 1e-12:
                continue
            
            h_test = current_h.copy()
            h_test[i_high] -= delta
            h_test[i_low] += delta
            h_test = np.maximum(h_test, 0.0)
            
            s_test = np.sum(h_test)
            if s_test < 1e-12:
                continue
            h_test = h_test * (N / s_test)
            
            val_test = compute_score(h_test)
            if val_test < current_val:
                current_h = h_test
                current_val = val_test
                improved = True
        
        if not improved:
            alpha *= 0.9
            if alpha < 1e-14:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try Gaussian-mode parametric optimization
    # params: [centers(n), sigmas(n), re_coeffs(n), im_coeffs(n)]
    for n_modes in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(100 + n_modes)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            for restart in range(8):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                # Random initialization
                centers = rng.uniform(-0.2, 0.2, size=n_modes)
                sigmas = rng.uniform(0.05, 0.15, size=n_modes)
                re_coeffs = rng.normal(0, 1.0, size=n_modes)
                im_coeffs = rng.normal(0, 1.0, size=n_modes)
                
                x0 = np.concatenate([centers, sigmas, re_coeffs, im_coeffs])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_modes, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x.copy()
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_modes)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Also try a simple cosine bump as baseline
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_cosine = np.cos(np.pi * x) ** 2
    f_cosine = np.maximum(f_cosine, 0)
    cos_score = compute_score(f_cosine)
    
    if cos_score < best_score:
        best_f = f_cosine
        best_score = cos_score
    
    # Local refinement: Fast peak shaving
    if time.time() - start_time < TIME_LIMIT - 15:
        refined_f, refined_score = fast_peak_shaving(
            best_f, start_time, TIME_LIMIT - 8
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # High-precision polish pass
    if time.time() - start_time < TIME_LIMIT - 3:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 1
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
