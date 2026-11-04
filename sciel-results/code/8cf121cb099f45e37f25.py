"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    N = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        # Shift 1-h by k positions
        shifted = np.roll(one_minus_h, k)
        # Overlap: sum of min(h[i], shifted[i]) * (2/N)
        ov = np.sum(np.minimum(h, shifted)) * (2.0 / N)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _compute_all_overlaps(h):
    """Compute overlap for all shifts, return array of length N."""
    N = len(h)
    one_minus_h = 1.0 - h
    overlaps = np.empty(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.sum(np.minimum(h, shifted)) * (2.0 / N)
    return overlaps

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    np.random.seed(42)
    
    # Start with a binary-like pattern: alternating 0s and 1s
    # This gives sum = N/2 automatically
    h = np.zeros(N)
    h[0::2] = 1.0
    h[1::2] = 0.0
    
    # Ensure sum is exactly N/2
    # For binary with N even, N/2 ones and N/2 zeros gives sum N/2
    
    best_h = h.copy()
    best_score = _compute_max_overlap(best_h)
    
    # Simulated annealing with swaps
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.9995
    max_iter = 500000
    
    current_score = best_score
    current_h = h.copy()
    
    # Precompute: for binary h, overlap at shift k is count of positions where
    # h[i]=1 and 1-h[(i-k)%N]=1, i.e., h[i]=1 and h[(i-k)%N]=0, plus
    # h[i]=0 and 1-h[(i-k)%N]=0, i.e., h[i]=0 and h[(i-k)%N]=1
    # So overlap = (number of 1-0 pairs at distance k + number of 0-1 pairs at distance k) * (2/N)
    # But since for binary, min(1, 0)=0, min(0,1)=0, min(1,1)=1, min(0,0)=0
    # Wait: min(h[i], 1-h[(i-k)%N])
    # If h[i]=1: min(1, 1-h[(i-k)%N]) = 1-h[(i-k)%N]
    # If h[i]=0: min(0, 1-h[(i-k)%N]) = 0
    # So overlap = sum over i where h[i]=1 of (1-h[(i-k)%N]) * (2/N)
    # = (number of i where h[i]=1 and h[(i-k)%N]=0) * (2/N)
    
    # For binary h, let's work with the binary representation
    # h[i] in {0,1}, sum = N/2
    
    # Faster overlap computation for binary
    def _binary_max_overlap(b):
        """b is binary array, return max overlap."""
        N = len(b)
        max_ov = 0.0
        for k in range(N):
            # Count positions where b[i]=1 and b[(i-k)%N]=0
            shifted = np.roll(b, k)
            cnt = np.sum(b * (1 - shifted))
            ov = cnt * (2.0 / N)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Recompute with binary method
    current_score = _binary_max_overlap(current_h)
    best_score = current_score
    best_h = current_h.copy()
    
    iter_count = 0
    while T > T_min and iter_count < max_iter:
        if time.time() > deadline:
            break
        
        # Pick two random positions to swap
        i, j = np.random.randint(0, N, 2)
        if i == j:
            continue
        if current_h[i] == current_h[j]:
            continue
        
        # Perform swap
        current_h[i], current_h[j] = current_h[j], current_h[i]
        
        new_score = _binary_max_overlap(current_h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            # Undo swap
            current_h[i], current_h[j] = current_h[j], current_h[i]
        
        T *= cooling_rate
        iter_count += 1
    
    # Local search: try all single swaps to see if any improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i+1, N):
                if current_h[i] == current_h[j]:
                    continue
                current_h[i], current_h[j] = current_h[j], current_h[i]
                new_score = _binary_max_overlap(current_h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_h = current_h.copy()
                else:
                    current_h[i], current_h[j] = current_h[j], current_h[i]
    
    # Final check
    final_score = _compute_max_overlap(best_h)
    return best_h
# EVOLVE-BLOCK-END
