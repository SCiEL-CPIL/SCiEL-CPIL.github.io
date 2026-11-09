"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = 140.0
    
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    
    def score_and_grad(h):
        h = np.maximum(h, 0)
        n = len(h)
        n_fft = 2 * n
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)
        total = np.sum(h)
        if total == 0:
            return 0.0, np.zeros(n)
        k_max = int(np.argmax(conv))
        s = conv[k_max] / (total * total)
        
        j = np.arange(n)
        idx1 = k_max - j
        idx2 = j - k_max
        grad = np.zeros(n)
        mask1 = (idx1 >= 0) & (idx1 < n)
        mask2 = (idx2 >= 0) & (idx2 < n)
        grad[mask1] += h[idx1[mask1]]
        grad[mask2] += h[idx2[mask2]]
        grad = grad / (total * total)
        
        return s, grad
    
    best_score, _ = score_and_grad(h)
    
    lr = 0.005
    for it in range(5000):
        if time.time() > deadline:
            break
        
        s, grad = score_and_grad(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        
        h = h - lr * grad
        h = np.maximum(h, 0)
        s_sum = np.sum(h)
        if s_sum > 1e-10:
            h = h * (np.sum(best_h) / s_sum)
    
    return list(best_h)
# EVOLVE-BLOCK-END
