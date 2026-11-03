# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_linear_overlaps(h):
    """
    Compute linear cross-correlation C(k) = sum_i h_i * (1 - h_{i+k})
    for k in [-(N-1), N-1].
    Returns array of length 2*N - 1.
    """
    N = len(h)
    # C(k) for k >= 0
    # C(k) = sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
    # We can compute this efficiently using convolution or loops.
    # Given N=100, loops are fine.
    C = np.zeros(2 * N - 1)
    # k = 0
    C[N-1] = np.sum(h * (1 - h))
    
    for k in range(1, N):
        # Positive lag k
        # i from 0 to N-1-k
        # term h[i] * (1 - h[i+k])
        val = np.sum(h[:N-k] * (1 - h[k:]))
        C[N-1+k] = val
        C[N-1-k] = val # Symmetry?
        # Is C(k) symmetric?
        # C(k) = sum h_i (1-h_{i+k})
        # C(-k) = sum h_i (1-h_{i-k})
        # Let j = i-k. C(-k) = sum h_{j+k} (1-h_j)
        # This is not necessarily equal to C(k).
        # However, sum h_i h_{i+k} is symmetric (autocorrelation).
        # C(k) = Sum(h_i in window) - A(k).
        # Sum(h_i in window) is symmetric (depends on N-k).
        # A(k) is symmetric.
        # So C(k) is symmetric.
        
    return C

def _max_overlap(h):
    C = _compute_linear_overlaps(h)
    return np.max(C)

def construct():
    N = 100
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial best: Constant solution
    h_best = np.full(N, 0.5)
    max_ov_best = _max_overlap(h_best) # Should be 25
    
    # Try random binary sequences
    # We want sum(h) = N/2 = 50
    for _ in range(200): # Up to 200 restarts
        if time.time() > deadline:
            break
            
        # Generate random binary sequence with 50 ones
        h = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1.0
        
        # Local search: try flipping bits to reduce max overlap
        # We can do a few iterations of local search
        current_max = _max_overlap(h)
        
        # Simple hill climbing
        for _ in range(50): # 50 steps of hill climbing
            if time.time() > deadline:
                break
            if current_max < max_ov_best:
                h_best = h.copy()
                max_ov_best = current_max
                
            # Pick a random bit to flip?
            # Or pick the bit that reduces max overlap most?
            # Just random flip for now
            i = np.random.randint(N)
            h[i] = 1.0 - h[i]
            
            # Check if sum is still N/2? No, flipping changes sum by 1.
            # We need to maintain sum = N/2.
            # So we must flip one 1 to 0 and one 0 to 1.
            # Let's do a swap move.
            h[i] = 1.0 - h[i] # Revert
            # Pick two indices, one with 1, one with 0
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            i1 = np.random.choice(ones)
            i0 = np.random.choice(zeros)
            
            h[i1] = 0.0
            h[i0] = 1.0
            
            new_max = _max_overlap(h)
            if new_max < current_max:
                current_max = new_max
            else:
                # Revert
                h[i1] = 1.0
                h[i0] = 0.0
                # Accept with some probability? (Simulated Annealing)
                # For now, strict hill climbing.
        
    return h_best
# EVOLVE-BLOCK-END
