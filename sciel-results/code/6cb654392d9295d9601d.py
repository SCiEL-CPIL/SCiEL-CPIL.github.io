"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search using FFT structure.
Key insight: autoconvolution = ifft(|FFT(f)|^2) (up to scaling).
This allows O(N log N) score evaluation instead of O(N^2).

We use iterative projection:
1. Start with a good initial shape (cosine bell, etc.)
2. Alternate between:
   a. Project to non-negative in time domain
   b. In frequency domain, modify |F|^2 to flatten the autoconvolution peak
   c. Take sqrt and re-construct f (maintaining non-negativity via projection)

This decouples the peak constraint from the time-domain shape search.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution using FFT."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # Autoconvolution via FFT: conv(f, f) = ifft(FFT(f) * FFT(f))
    # But for linear convolution, we need to zero-pad
    F = np.fft.fft(f, n=2*N)
    autoconv = np.fft.ifft(F * F).real[:2*N-1]
    
    # The full linear convolution has length 2*N-1
    # Normalize by (sum(f))^2
    return np.max(autoconv) / (s * s)

def compute_score_fft_only(f):
    """Faster version that assumes f is already non-negative."""
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    F = np.fft.fft(f, n=2*N)
    autoconv = np.fft.ifft(F * F).real[:2*N-1]
    return np.max(autoconv) / (s * s)

def cosine_bell(n, width_ratio=0.8):
    """Create a cosine bell shape centered in the array."""
    f = np.zeros(n)
    center = n // 2
    half_width = int(n * width_ratio / 2)
    if half_width < 1:
        f[center] = 1.0
        return f
    for i in range(-half_width, half_width + 1):
        x = i / half_width
        if abs(x) <= 1.0:
            f[center + i] = 0.5 * (1 + np.cos(np.pi * x))
    return f

def random_bell(n, center_frac=None, width_frac=None):
    """Create a random bell-shaped function."""
    if center_frac is None:
        center_frac = np.random.uniform(0.3, 0.7)
    if width_frac is None:
        width_frac = np.random.uniform(0.3, 0.9)
    
    f = np.zeros(n)
    center = int(n * center_frac)
    half_width = max(1, int(n * width_frac / 2))
    
    for i in range(-half_width, half_width + 1):
        idx = center + i
        if 0 <= idx < n:
            x = i / half_width
            if abs(x) <= 1.0:
                f[idx] = 0.5 * (1 + np.cos(np.pi * x))
    
    # Add some random perturbation
    if np.random.rand() < 0.5:
        noise_scale = np.random.uniform(0.0, 0.1)
        f += noise_scale * np.random.randn(n)
        f = np.maximum(f, 0.0)
    
    return f

def iterative_projection(f, n_iter=5000, time_budget=None, start_time=None):
    """
    Iterative projection algorithm to minimize peak of autoconvolution.
    """
    if time_budget is None:
        time_budget = 10.0
    if start_time is None:
        start_time = time.time()
    
    f = np.asarray(f, dtype=np.float64).copy()
    f = np.maximum(f, 0.0)
    
    best_f = f.copy()
    best_score = compute_score_fft_only(f)
    
    for it in range(n_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Random perturbation
        if np.random.rand() < 0.3:
            perturbation_scale = np.random.uniform(0.01, 0.1)
            f_new = f + perturbation_scale * np.random.randn(N)
            f_new = np.maximum(f_new, 0.0)
            
            s_new = np.sum(f_new)
            if s_new > 1e-15:
                f_new /= s_new * np.sum(f)
            
            score_new = compute_score_fft_only(f_new)
            if score_new < best_score:
                best_score = score_new
                best_f = f_new.copy()
                f = f_new.copy()
        
        # Smoothing
        if np.random.rand() < 0.2:
            kernel = np.ones(3) / 3.0
            f_smooth = np.convolve(f, kernel, mode='same')
            f_smooth = np.maximum(f_smooth, 0.0)
            s_smooth = np.sum(f_smooth)
            if s_smooth > 1e-15:
                f_smooth /= s_smooth * np.sum(f)
            
            score_smooth = compute_score_fft_only(f_smooth)
            if score_smooth < best_score:
                best_score = score_smooth
                best_f = f_smooth.copy()
                f = f_smooth.copy()
        
        # Sharpening
        if np.random.rand() < 0.2:
            f_sharp = f ** 1.5
            s_sharp = np.sum(f_sharp)
            if s_sharp > 1e-15:
                f_sharp /= s_sharp * np.sum(f)
            
            score_sharp = compute_score_fft_only(f_sharp)
            if score_sharp < best_score:
                best_score = score_sharp
                best_f = f_sharp.copy()
                f = f_sharp.copy()
    
    return best_f, best_score

def spectral_flattening(f, n_iter=10000, time_budget=10.0, start_time=None):
    """
    Spectral flattening algorithm.
    """
    if start_time is None:
        start_time = time.time()
    
    f = np.asarray(f, dtype=np.float64).copy()
    f = np.maximum(f, 0.0)
    
    best_f = f.copy()
    best_score = compute_score_fft_only(f)
    
    fft_size = 2 * N
    
    for it in range(n_iter):
        if time.time() - start_time > time_budget:
            break
        
        F = np.fft.fft(f, n=fft_size)
        mag2 = np.abs(F) ** 2
        
        mean_mag2 = np.mean(mag2)
        
        alpha = 0.95
        mag2_flat = alpha * mag2 + (1 - alpha) * mean_mag2
        mag2_flat = np.maximum(mag2_flat, 0)
        
        phase = np.angle(F)
        F_new = np.sqrt(mag2_flat) * np.exp(1j * phase)
        
        f_new = np.fft.ifft(F_new).real[:N]
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        s_old = np.sum(f)
        if s_new > 1e-15 and s_old > 1e-15:
            f_new = f_new * (s_old / s_new)
        
        score_new = compute_score_fft_only(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            f = f_new.copy()
        else:
            if np.random.rand() < 0.5:
                perturbation_scale = np.random.uniform(0.001, 0.01)
                f_perturbed = f + perturbation_scale * np.random.randn(N)
                f_perturbed = np.maximum(f_perturbed, 0.0)
                s_perturbed = np.sum(f_perturbed)
                if s_perturbed > 1e-15:
                    f_perturbed = f_perturbed * (s_old / s_perturbed)
                
                score_perturbed = compute_score_fft_only(f_perturbed)
                if score_perturbed < best_score:
                    best_score = score_perturbed
                    best_f = f_perturbed.copy()
                    f = f_perturbed.copy()
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    best_f = cosine_bell(N, width_ratio=0.9)
    s_init = np.sum(best_f)
    if s_init > 1e-15:
        best_f /= s_init
    best_score = compute_score_fft_only(best_f)
    
    initializations = [
        cosine_bell(N, width_ratio=0.5),
        cosine_bell(N, width_ratio=0.7),
        cosine_bell(N, width_ratio=0.9),
        random_bell(N, 0.5, 0.4),
        random_bell(N, 0.5, 0.6),
        random_bell(N, 0.5, 0.8),
    ]
    
    uniform_f = np.ones(N) / N
    initializations.append(uniform_f)
    
    tri_f = np.zeros(N)
    for i in range(N):
        x = i / (N - 1)
        tri_f[i] = 1 - abs(2 * x - 1)
    initializations.append(tri_f)
    
    for init_f in initializations:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        init_f = np.asarray(init_f, dtype=np.float64)
        init_f = np.maximum(init_f, 0.0)
        s_init = np.sum(init_f)
        if s_init > 1e-15:
            init_f = init_f / s_init
        
        f_opt, score_opt = spectral_flattening(init_f, n_iter=20000, time_budget=TIME_LIMIT - 15, start_time=start_time)
        
        if score_opt < best_score:
            best_score = score_opt
            best_f = f_opt.copy()
        
        if time.time() - start_time < TIME_LIMIT - 10:
            f_opt2, score_opt2 = iterative_projection(f_opt, n_iter=50000, time_budget=TIME_LIMIT - 10, start_time=start_time)
            if score_opt2 < best_score:
                best_score = score_opt2
                best_f = f_opt2.copy()
    
    if time.time() - start_time < TIME_LIMIT - 5:
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            perturbation_scale = np.random.uniform(0.0001, 0.001)
            f_perturbed = best_f + perturbation_scale * np.random.randn(N)
            f_perturbed = np.maximum(f_perturbed, 0.0)
            s_perturbed = np.sum(f_perturbed)
            s_best = np.sum(best_f)
            if s_perturbed > 1e-15 and s_best > 1e-15:
                f_perturbed = f_perturbed * (s_best / s_perturbed)
            
            score_perturbed = compute_score_fft_only(f_perturbed)
            if score_perturbed < best_score:
                best_score = score_perturbed
                best_f = f_perturbed.copy()
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
