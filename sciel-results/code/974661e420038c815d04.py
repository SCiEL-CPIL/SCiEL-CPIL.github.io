"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a spectral/parametric approach with cosine basis functions,
then refine with local optimization. The key insight is that flat autoconvolution
corresponds to a flat power spectrum, so we start with a spectrum-flattening
approach and then do local peak-shaving.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def fft_convolve(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.fft(f, fft_size)
    C = np.fft.ifft(F * F)
    return C.real[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_convolve(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(conv, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    max_val = np.max(conv)
    if max_val < 1e-15:
        return max_val
    shifted = conv / temperature
    shifted = shifted - np.max(shifted)
    return temperature * (np.max(conv) + np.log(np.sum(np.exp(shifted))))

def cosine_basis(x, k):
    """Cosine basis function for k-th mode."""
    return np.cos(k * np.pi * x)

def parametric_to_step(params, x):
    """Convert parametric coefficients to step function values."""
    f = np.zeros_like(x)
    for i, coeff in enumerate(params):
        f += coeff * cosine_basis(x, i)
    # Ensure nonnegativity
    f = np.maximum(f, 0.0)
    return f

def objective(params, x):
    """Objective: normalized max of autoconvolution."""
    f = parametric_to_step(params, x)
    return compute_score(f)

def compute_sensitivity_fft(f):
    """Compute sensitivity of autoconvolution peaks to changes in f."""
    conv = fft_convolve(f)
    n = len(f)
    # Find peak indices
    max_val = np.max(conv)
    threshold = max_val * 0.95
    peak_indices = np.where(conv >= threshold)[0]
    
    # Compute gradient-like sensitivity
    sens = np.zeros(n)
    for m in peak_indices:
        for i in range(n):
            if 0 <= m - i < n:
                sens[i] += f[m - i]
            if 0 <= i - m < n:
                sens[i] += f[i - m]
    return sens

def local_refinement(f, start_time, time_budget=30):
    """Local refinement using subgradient descent with adaptive step size."""
    alpha = 0.03
    patience = 40
    no_improve = 0
    best_f = f.copy()
    best_score = compute_score(f)
    
    while time.time() - start_time < TIME_LIMIT - time_budget:
        conv = fft_convolve(f)
        max_val = np.max(conv)
        
        # Find lags with near-max value
        threshold = max_val * 0.98
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute subgradient
        g = np.zeros(N)
        for m in peak_indices:
            for i in range(N):
                if 0 <= m - i < N:
                    g[i] += f[m - i]
                if 0 <= i - m < N:
                    g[i] += f[i - m]
        
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        g = g / g_norm
        
        # Update with projection onto nonnegative simplex
        f_new = f - alpha * g
        f_new = np.maximum(f_new, 0.0)
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            break
        f_new = f_new / s_new
        
        score_new = compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            no_improve = 0
            alpha = min(alpha * 1.02, 0.15)
        else:
            no_improve += 1
            alpha *= 0.92
            if no_improve > patience:
                break
            f = f_new
    
    return best_f, best_score

def polish_pass(f, start_time, time_budget=15):
    """Polish using coordinate descent on individual bins."""
    current_score = compute_score(f)
    best_f = f.copy()
    
    max_passes = 5
    for _ in range(max_passes):
        if time.time() - start_time > TIME_LIMIT - time_budget:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > TIME_LIMIT - time_budget:
                break
            
            f_orig = f[i]
            max_f = np.max(f)
            upper_bound = 2 * max_f + 1e-6
            
            # Ternary search for optimal f[i]
            a = 0.0
            b = upper_bound
            for _ in range(15):
                m1 = a + (b - a) / 3
                m2 = b - (b - a) / 3
                
                f[i] = m1
                s = np.sum(f)
                if s < 1e-15:
                    continue
                f_norm = f / s
                score1 = compute_score(f_norm)
                
                f[i] = m2
                s = np.sum(f)
                if s < 1e-15:
                    continue
                f_norm = f / s
                score2 = compute_score(f_norm)
                
                if score1 < score2:
                    b = m2
                else:
                    a = m1
            
            f[i] = (a + b) / 2
            s = np.sum(f)
            if s < 1e-15:
                break
            f = f / s
            score_final = compute_score(f)
            
            if score_final < current_score:
                current_score = score_final
                best_f = f.copy()
                improved = True
            else:
                f[i] = f_orig
                s = np.sum(f)
                if s > 1e-15:
                    f = f / s
        
        if not improved:
            break
    
    return best_f, current_score

def construct():
    start_time = time.time()
    
    # Initialize with a smooth bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    sigma = 0.09
    f = np.exp(-0.5 * (x / sigma) ** 2)
    f = f / np.sum(f)
    
    best_f = f.copy()
    best_score = compute_score(f)
    
    # Phase 1: Local refinement with subgradient descent
    if time.time() - start_time < TIME_LIMIT - 40:
        f, score = local_refinement(f, start_time, time_budget=30)
        if score < best_score:
            best_score = score
            best_f = f
    
    # Phase 2: Coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 5:
        f, score = polish_pass(best_f, start_time, time_budget=15)
        if score < best_score:
            best_score = score
            best_f = f
    
    # Final normalization
    s_final = np.sum(best_f)
    if s_final > 1e-15:
        best_f = best_f / s_final
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
