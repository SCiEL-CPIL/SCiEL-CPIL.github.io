import numpy as np
import time

N = 100

def _autocorr_min(h):
    """Compute the minimum autocorrelation over all non-zero shifts."""
    h = np.asarray(h)
    min_ac = float('inf')
    for k in range(1, N):
        ac = np.sum(h * np.roll(h, k))
        if ac < min_ac:
            min_ac = ac
    return min_ac

def _overlap_max(h):
    """Compute the maximum overlap over all non-zero shifts."""
    h = np.asarray(h)
    max_ov = 0.0
    for k in range(1, N):
        ov = np.sum(h * (1 - np.roll(h, k)))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)  # should be 25.0
    
    # Binary local search
    np.random.seed(42)
    
    # Start with a structured binary sequence
    h = np.zeros(N)
    # Place ones at even indices initially
    h[::2] = 1.0
    
    # Local search: try swapping a 1 and a 0
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    current_min_ac = _autocorr_min(h)
    current_max_ov = _overlap_max(h)
    
    if current_max_ov < best_score:
        best_score = current_max_ov
        best_h = h.copy()
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5.0:
        if no_improve_count > max_no_improve:
            # Restart with new random structure
            np.random.seed(int(time.time() * 1000) % (2**31))
            h = np.zeros(N)
            indices = np.random.choice(N, N//2, replace=False)
            h[indices] = 1.0
            current_min_ac = _autocorr_min(h)
            current_max_ov = _overlap_max(h)
            if current_max_ov < best_score:
                best_score = current_max_ov
                best_h = h.copy()
            no_improve_count = 0
            max_no_improve = 500
            continue
        
        # Pick a random 1 and a random 0 to swap
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        if i == j:
            continue
        
        # Try the swap
        h[i] = 0
        h[j] = 1
        
        new_max_ov = _overlap_max(h)
        
        if new_max_ov < current_max_ov:
            # Accept
            current_max_ov = new_max_ov
            # Update ones and zeros
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            no_improve_count = 0
            if current_max_ov < best_score:
                best_score = current_max_ov
                best_h = h.copy()
        else:
            # Revert
            h[i] = 1
            h[j] = 0
            no_improve_count += 1
    
    # Return the best found
    return best_h
