import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Strategy: Use a mixture of K Gaussians centered at non-adjacent indices
    # to minimize the peak of the autoconvolution while maintaining high L2 norm.
    # Optimal K for non-overlapping support is approx N/2.
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try different numbers of spikes K
    # K=25 is the max for non-overlapping (indices 0, 2, ..., 48)
    # K=24 (indices 1, 3, ..., 47) etc.
    
    for K in [25, 24, 20, 10, 5]:
        # Determine centers
        if K == 25:
            centers = np.arange(0, N, 2)
        elif K == 24:
            centers = np.arange(1, N, 2)
        elif K == 20:
            centers = np.linspace(0, N-1, K)
        elif K == 10:
            centers = np.linspace(0, N-1, K)
        else:
            centers = np.linspace(0, N-1, K)
            
        # Search for optimal width sigma
        # Small sigma is better, but must be > 0 to be valid floating point
        # and to allow the "Gaussian" structure.
        # We try a range of sigmas, prioritizing very small ones.
        
        for sigma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5]:
            h = np.zeros(N)
            x = np.arange(N)
            
            for c in centers:
                # Add Gaussian contribution
                h += np.exp(-0.5 * ((x - c) / sigma)**2)
            
            l1 = np.sum(h)
            if l1 > 0:
                h_norm = h / l1
                s_candidate = _score_raw(h_norm)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_norm.copy()
                    
    return list(best_h)
