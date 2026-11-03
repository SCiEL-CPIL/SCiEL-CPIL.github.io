"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Start from a well-chosen initial guess (e.g., cosine bump or flat).
2. Use projected gradient descent with FFT-based autoconvolution.
3. Apply adaptive step sizes and nonnegativity projection.
4. Use peak-aware sensitivity to guide updates.
5. Final polish with smaller steps.
"""

import numpy as np
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
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
    return conv

def initial_guess_cosine():
    """Cosine bump centered at 0."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f = np.cos(np.pi * x / 0.25) ** 2
    f = np.maximum(f, 0.0)
    return f

def initial_guess_flat():
    """Flat function."""
    return np.ones(N)

def initial_guess_gaussian():
    """Gaussian bump."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f = np.exp(-20 * x ** 2)
    return f

def projected_gradient_descent(f_init, start_time, max_time, lr_init=0.05, lr_min=1e-6):
    """Projected gradient descent with FFT-based autoconvolution."""
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    lr = lr_init
    
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    while time.time() - start_time < max_time:
        # Normalize to maintain sum
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        # Compute autoconvolution
        conv = fft_autocorr(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Compute gradient: d/d f_i of max(conv)
        # conv[k] = sum_j f[j] * f[k-j]
        # d conv[k] / d f[i] = f[k-i] + f[i-k] (with appropriate bounds)
        # For peak reduction, we want to reduce f[i] where it contributes to peaks
        
        # Find peak indices
        threshold = max_c * 0.99
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        # Compute sensitivity: how much each f[i] contributes to peak autoconvolution
        # Use FFT cross-correlation
        peak_ind = np.zeros(fft_size)
        peak_ind[peak_indices] = 1.0
        
        f_padded = np.zeros(fft_size)
        f_padded[:N] = current_f
        
        F = np.fft.rfft(f_padded)
        IP = np.fft.rfft(peak_ind)
        S = np.fft.irfft(F * np.conj(IP), n=fft_size)
        sensitivity = S[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Update: reduce f where sensitivity is high
        f_new = current_f * (1.0 - lr * sens_norm)
        f_new = np.maximum(f_new, 0.0)
        
        # Normalize
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        # Check improvement
        conv_new = fft_autocorr(f_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            lr = min(lr * 1.02, 0.2)
        else:
            lr *= 0.95
            if lr < lr_min:
                break
    
    return best_f, best_val

def polish_pass(f_init, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    lr = 0.005
    
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autocorr(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        threshold = max_c * 0.999
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        peak_ind = np.zeros(fft_size)
        peak_ind[peak_indices] = 1.0
        
        f_padded = np.zeros(fft_size)
        f_padded[:N] = current_f
        
        F = np.fft.rfft(f_padded)
        IP = np.fft.rfft(peak_ind)
        S = np.fft.irfft(F * np.conj(IP), n=fft_size)
        sensitivity = S[:N]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        f_new = current_f * (1.0 - lr * sens_norm)
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        conv_new = fft_autocorr(f_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            lr = min(lr * 1.01, 0.05)
        else:
            lr *= 0.98
            if lr < 1e-12:
                break
    
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Try multiple initializations
    inits = [
        initial_guess_flat(),
        initial_guess_cosine(),
        initial_guess_gaussian(),
    ]
    
    best_f = None
    best_score = np.inf
    
    for f_init in inits:
        f_init = np.maximum(f_init, 0.0)
        s = np.sum(f_init)
        if s < 1e-12:
            continue
        f_init = f_init * (N / s)
        score = compute_score(f_init)
        if score < best_score:
            best_score = score
            best_f = f_init.copy()
    
    if best_f is None:
        best_f = np.ones(N)
        best_score = compute_score(best_f)
    
    # Projected gradient descent
    if time.time() - start_time < TIME_LIMIT - 20:
        pgd_f, pgd_score = projected_gradient_descent(
            best_f, start_time, TIME_LIMIT - 15
        )
        if pgd_score < best_score:
            best_f = pgd_f
            best_score = pgd_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 1:
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
