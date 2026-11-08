import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def compute_ratio(h):
        """Compute ||h||^2 / (||h||_1 * ||h*h||_inf) for step heights."""
        h = np.array(h, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k] for k>=0
        # For nonnegative h, the peak is at lag 0
        # But let's compute properly
        autoconv = np.correlate(h, h, mode='full')
        sup_ac = np.max(autoconv)
        if sup_ac < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_ac)
    
    def make_block(k, weights, center=True):
        """Create a 50-element array with a block of k nonzero values."""
        h = np.zeros(N)
        if center:
            start = (N - k) // 2
        else:
            start = 0
        for i in range(k):
            h[start + i] = weights[i]
        return h
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_ratio = compute_ratio(best_h)
    
    # Try different block sizes and optimize weights
    for k in range(2, 16):
        if time.time() > deadline:
            break
        
        # Start with equal weights
        weights = np.ones(k) / k
        h = make_block(k, weights)
        ratio = compute_ratio(h)
        
        # Local search: try perturbing weights
        # Use a simple gradient-free approach: try moving mass between adjacent elements
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(k - 1):
                # Try moving small amount from i to i+1 and vice versa
                for delta in [0.01, 0.05, 0.1, 0.2]:
                    w_new = weights.copy()
                    w_new[i] -= delta
                    w_new[i+1] += delta
                    if w_new[i] < 0 or w_new[i+1] < 0:
                        continue
                    # Normalize
                    s = np.sum(w_new)
                    if s < 1e-15:
                        continue
                    w_new = w_new / s
                    h_new = make_block(k, w_new)
                    r_new = compute_ratio(h_new)
                    if r_new > ratio + 1e-10:
                        ratio = r_new
                        weights = w_new
                        improved = True
                        break
                    # Also try reverse
                    w_new2 = weights.copy()
                    w_new2[i] += delta
                    w_new2[i+1] -= delta
                    if w_new2[i] < 0 or w_new2[i+1] < 0:
                        continue
                    s2 = np.sum(w_new2)
                    if s2 < 1e-15:
                        continue
                    w_new2 = w_new2 / s2
                    h_new2 = make_block(k, w_new2)
                    r_new2 = compute_ratio(h_new2)
                    if r_new2 > ratio + 1e-10:
                        ratio = r_new2
                        weights = w_new2
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = make_block(k, weights)
    
    # Also try non-contiguous patterns: e.g., two separate blocks
    for k1 in range(1, 8):
        for k2 in range(1, 8):
            if time.time() > deadline:
                break
            for gap in range(1, 6):
                total = k1 + gap + k2
                if total > N:
                    continue
                # Two blocks with weights
                w1 = np.ones(k1) / k1
                w2 = np.ones(k2) / k2
                # Optimize split
                for split in np.linspace(0.1, 0.9, 10):
                    w1 = w1 * split
                    w2 = w2 * (1 - split)
                    h = np.zeros(N)
                    start1 = (N - total) // 2
                    for i in range(k1):
                        h[start1 + i] = w1[i]
                    for i in range(k2):
                        h[start1 + k1 + gap + i] = w2[i]
                    r = compute_ratio(h)
                    if r > best_ratio:
                        best_ratio = r
                        best_h = h
    
    # Final: ensure nonnegativity and return list
    best_h = np.maximum(best_h, 0)
    return list(best_h)
# EVOLVE-BLOCK-END
