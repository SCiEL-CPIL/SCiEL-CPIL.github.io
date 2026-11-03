import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115

def _compute_max_overlap(h):
    """
    Compute the maximum linear overlap C_k for the given h.
    h is an array of 0s and 1s (or floats).
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    # C_k = (2/N) * (N/2 - r_k)
    cs = (2.0 / n) * (n / 2.0 - r)
    return float(np.max(cs))

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    # Initial state: 50 ones, 50 zeros
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N//2, replace=False)
    h[ones_indices] = 1.0
    
    best_h = h.copy()
    best_score = _compute_max_overlap(h)
    
    # Simulated Annealing parameters
    max_iters = 1000000
    T_start = 1.0
    T_end = 0.001
    
    current_score = best_score
    
    for it in range(max_iters):
        if time.time() - t0 > _DEADLINE - 5:
            break
            
        # Temperature
        progress = it / max_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Pick two random indices to swap
        # Ensure they are different values
        # Since we have exactly 50 ones and 50 zeros,
        # we can pick a random 1 and a random 0.
        ones_mask = (h == 1.0)
        zeros_mask = (h == 0.0)
        
        # Get indices
        ones_idx = np.where(ones_mask)[0]
        zeros_idx = np.where(zeros_mask)[0]
        
        i = np.random.choice(ones_idx)
        j = np.random.choice(zeros_idx)
        
        # Swap
        h[i] = 0.0
        h[j] = 1.0
        
        new_score = _compute_max_overlap(h)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Reject, swap back
            h[i] = 1.0
            h[j] = 0.0
            
    return best_h
