"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import numpy as np
from scipy.optimize import minimize
import time

def _score(h):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
    h = np.array(h, dtype=float)
    n = len(h)
    w = 1.0 / (2.0 * n)  # interval width: [-1/4, 1/4] divided into n intervals
    
    # L2^2 and L1
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    # Autoconvolution
    # Use a grid fine enough to resolve the spike
    n_grid = 20001
    t = np.linspace(-0.5, 0.5, n_grid)
    dt = t[1] - t[0]
    
    centers = -0.25 + (np.arange(n) + 0.5) * w
    f_grid = np.zeros(n_grid)
    # Vectorized assignment for speed
    for i in range(n):
        lo_idx = int(np.searchsorted(t, centers[i] - w/2))
        hi_idx = int(np.searchsorted(t, centers[i] + w/2))
        if hi_idx > lo_idx:
            f_grid[lo_idx:hi_idx] += h[i]
            
    # Convolve using FFT
    # Zero pad to avoid circular convolution artifacts
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Candidate 1: Uniform
    h_uniform = np.ones(N)
    
    # Candidate 2: Single Spike (Center)
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    
    # Candidate 3: Two Spikes
    h_two_spikes = np.zeros(N)
    h_two_spikes[N//4] = 1.0
    h_two_spikes[3*N//4] = 1.0
    
    # Candidate 4: Triangular
    h_tri = np.zeros(N)
    for i in range(N):
        h_tri[i] = 1.0 - abs(i - N//2) / (N//2)
    h_tri[h_tri < 0] = 0
    
    # Evaluate candidates
    candidates = [h_uniform, h_spike, h_two_spikes, h_tri]
    best_h = h_uniform
    best_score = _score(h_uniform)
    
    for h in candidates[1:]:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # Local optimization
    if time.time() < deadline - 5:
        def neg_score(h):
            h = np.abs(h)
            return -_score(h)
        
        # Use Nelder-Mead with adaptive settings
        result = minimize(neg_score, best_h, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.success or -result.fun > best_score:
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    
    return list(best_h)
# EVOLVE-BLOCK-END
