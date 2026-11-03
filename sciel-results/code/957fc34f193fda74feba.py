"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a spectral parameterization where the step function is represented
as the squared magnitude of a complex signal, ensuring non-negativity. We optimize
the real and imaginary parts of the spectral coefficients to flatten the peak of
the autoconvolution. This is combined with a time-domain local refinement pass
using FFT-based convolution and sensitivity-based peak shaving.

The key insight is that by working in the spectral domain with a |sum of modes|^2
parameterization, we can use smooth optimization (Nelder-Mead) on a lower-dimensional
space that inherently satisfies the non-negativity constraint, then refine in the
time domain with a fast gradient-like method.
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

def spectral_to_step(params, n_modes):
    """Convert spectral parameters to a nonnegative step function.
    
    params: [real_0, imag_0, real_1, imag_1, ..., real_{n-1}, imag_{n-1}]
    The step function is |sum_k (a_k + i*b_k) * exp(i * omega_k * x)|^2
    where omega_k are chosen frequencies.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Choose frequencies: use a set of harmonics
    # We'll use frequencies that are multiples of a base frequency
    base_freq = 2.0  # cycles per unit interval
    freqs = np.arange(n_modes) * base_freq + base_freq / 2.0
    
    # Complex coefficients
    reals = params[0::2][:n_modes]
    imags = params[1::2][:n_modes]
    
    # Compute the complex signal
    signal = np.zeros(N, dtype=np.complex128)
    for k in range(n_modes):
        signal += (reals[k] + 1j * imags[k]) * np.exp(1j * freqs[k] * 2 * np.pi * x)
    
    # Squared magnitude
    f = np.abs(signal) ** 2
    f = np.maximum(f, 0.0)
    return f

def objective_spectral(params, n_modes, beta=10.0):
    """Objective for spectral optimization."""
    f = spectral_to_step(params, n_modes)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def fast_peak_shaving(h, start_time, max_time):
    """Fast local refinement using FFT-based convolution and gradient-like steps."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    patience = 0
    max_patience = 300
    step_count = 0
    
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
        
        if current_val < best_val - 1e-15:
            best_val = current_val
            best_h = current_h.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find the index of the maximum
        k_star = np.argmax(conv)
        
        # Compute gradient direction
        g = np.zeros(N)
        j_min = max(0, k_star - (N - 1))
        j_max = min(N - 1, k_star)
        
        if j_min <= j_max:
            indices = np.arange(j_min, j_max + 1)
            f_indices = k_star - indices
            g[indices] = current_h[f_indices]
            
            g_mean = np.mean(g)
            g = g - g_mean
            
            h_new = current_h - alpha * g
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            val_new = compute_score(h_new)
            
            if val_new < current_val - 1e-15:
                current_h = h_new
                alpha = min(alpha * 1.05, 0.5)
                step_count += 1
            else:
                alpha *= 0.9
                if alpha < 1e-8:
                    break
    
    return best_h, best_val

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

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Strategy 1: Spectral optimization
    for n_modes in [3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(100 + n_modes)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                # Random initialization
                params0 = rng.normal(0, 1, size=2*n_modes)
                
                try:
                    result = minimize(
                        objective_spectral, params0, args=(n_modes, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = spectral_to_step(best_params, n_modes)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Strategy 2: Cosine bump parametric optimization
    for n_c in [2, 3, 4]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        for beta in [5.0, 10.0]:
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8}
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
    
    # Local refinement: Fast peak shaving
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = fast_peak_shaving(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Also try a simple cosine bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_cosine = np.cos(np.pi * x) ** 2
    f_cosine = np.maximum(f_cosine, 0)
    cos_score = compute_score(f_cosine)
    
    if cos_score < best_score and time.time() - start_time < TIME_LIMIT - 20:
        best_f = f_cosine
        best_score = cos_score
        refined_cos, refined_cos_score = fast_peak_shaving(
            f_cosine, start_time, TIME_LIMIT - 5
        )
        if refined_cos_score < best_score:
            best_f = refined_cos
            best_score = refined_cos_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
