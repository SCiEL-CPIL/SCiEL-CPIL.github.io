"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Start with a random feasible initialization
    np.random.seed(42)
    h = np.random.rand(N)
    # Project to sum = N/2
    h = h - h.mean() + N / (2.0 * N)
    h = np.clip(h, 0, 1)
    # Re-project to ensure exact sum
    diff = N / 2.0 - h.sum()
    h += diff / N
    h = np.clip(h, 0, 1)
    
    best_h = h.copy()
    best_score = float('inf')
    
    def compute_max_overlap(h):
        """Compute the maximum overlap of h with shifted 1-h over all non-zero shifts."""
        h = np.asarray(h)
        n = len(h)
        max_ov = 0.0
        for k in range(1, n):
            shifted = np.roll(h, -k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    best_score = compute_max_overlap(h)
    
    # Use a simple iterative refinement: try to reduce the worst overlap
    # by adjusting elements
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute overlaps for all shifts
        h_curr = h.copy()
        overlaps = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(h_curr, -k)
            overlaps[k] = np.sum(h_curr * (1.0 - shifted))
        
        # Find the worst shift
        worst_k = np.argmax(overlaps[1:]) + 1
        worst_ov = overlaps[worst_k]
        
        if worst_ov < best_score - 1e-10:
            best_score = worst_ov
            best_h = h_curr.copy()
        
        # Compute gradient-like adjustment: to reduce overlap at shift k,
        # we want to decrease h[i] where (1-h[i+k]) is high, or increase where it's low
        shifted = np.roll(h_curr, -worst_k)
        # overlap = sum h[i]*(1-h[i+k])
        # d(overlap)/d(h[i]) = (1 - h[i+k]) + h[i-k] * (-1) [from the term where h[i] appears as the shifted one]
        # Actually: overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # d/dh[i] = (1 - h[(i+k)%N]) + h[(i-k)%N] * (-1) * ... wait
        # h[i] appears in term i as h[i]*(1-h[i+k]) and in term (i-k) as h[i-k]*(1-h[i])
        # So d/dh[i] = (1 - h[(i+k)%N]) - h[(i-k)%N]
        
        grad = (1.0 - shifted) - np.roll(h_curr, worst_k)
        
        # Update: decrease h where gradient is positive (to reduce overlap)
        h_new = h_curr - lr * grad
        h_new = np.clip(h_new, 0, 1)
        
        # Re-project to sum = N/2
        diff = N / 2.0 - h_new.sum()
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        
        h = h_new
    
    # Final projection to ensure exact sum
    diff = N / 2.0 - best_h.sum()
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
