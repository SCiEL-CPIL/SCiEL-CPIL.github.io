import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum autoconvolution score.
    Score = max_k S_k / S_0, where S_k = sum_i h_i h_{i+k}
    """
    H = np.sum(h)
    if H < 1e-12:
        return np.inf
    
    # Compute autoconvolution coefficients S_k using FFT for efficiency
    n_fft = 2 * N
    H_fft = np.fft.rfft(h, n=n_fft)
    S_fft = H_fft * np.conj(H_fft)
    S = np.fft.irfft(S_fft, n=n_fft)[:N]
    
    S_0 = S[0]
    if S_0 < 1e-12:
        return np.inf
    
    return np.max(S) / S_0

def smooth_max(S, S_0, alpha=100.0):
    """
    Smooth surrogate for max(S)/S_0 using log-sum-exp.
    This is differentiable and approximates the max when alpha is large.
    """
    if S_0 < 1e-12:
        return 1e10
    x = S / S_0
    x = np.clip(x, 1e-10, 1e10)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Try various structured patterns that might concentrate autoconvolution
    x = np.linspace(-1, 1, N)
    
    patterns = []
    
    # Gaussian-like peaks with different widths
    for sigma in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        patterns.append(np.exp(-x**2 / (2 * sigma**2)))
    
    # Cosine bumps
    for freq in [1, 2, 3]:
        patterns.append(0.5 * (1 + np.cos(freq * np.pi * x)))
    
    # Lorentzian peaks
    patterns.append(1 / (1 + 10 * x**2))
    patterns.append(1 / (1 + 5 * x**2))
    patterns.append(1 / (1 + 2 * x**2))
    
    # Power law peaks
    for p in [2, 3, 4, 5]:
        patterns.append(np.maximum(0, 1 - np.abs(x)**p))
    
    # Triangle peak
    patterns.append(np.maximum(0, 1 - 2 * np.abs(x)))
    
    # Evaluate all patterns
    for p in patterns:
        if np.min(p) < 0:
            p = np.clip(p, 0, None)
        if np.sum(p) < 1e-12:
            continue
        s = compute_score(p)
        if s < best_score:
            best_score = s
            best_h = p.copy()
    
    # If we found something better than 1, try to optimize it further
    if best_score < 0.999 and time.time() < deadline - 30:
        h0 = best_h.copy()
        
        def objective(params):
            h = np.exp(params)
            if time.time() > deadline - 10:
                return 1e10
            n_fft = 2 * N
            H_fft = np.fft.rfft(h, n=n_fft)
            S_fft = H_fft * np.conj(H_fft)
            S = np.fft.irfft(S_fft, n=n_fft)[:N]
            S_0 = S[0]
            if S_0 < 1e-12:
                return 1e10
            return smooth_max(S, S_0, alpha=50.0)
        
        try:
            result = minimize(objective, np.log(h0 + 1e-10), method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_score:
                h_opt = np.exp(result.x)
                s_opt = compute_score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    return list(best_h)

construct()
