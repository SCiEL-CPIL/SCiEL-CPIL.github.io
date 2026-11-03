"""Rewritten program for the suite task."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _max_overlap(h):
    """
    Calculate the maximum overlap of h with shifted 1-h.
    Overlap for shift k is sum(h[i] * (1 - h[i+k])) for i in 0..N-1-k.
    """
    max_val = 0.0
    # We only need to check shifts that could potentially be max.
    # For uniform, max is at k=0.
    # For binary, k=0 is 0. Max is likely at small k.
    # But to be safe, check all.
    # Optimization: N=100, 100 shifts, 100 ops each = 10000 ops. Very fast.
    for k in range(N):
        overlap = 0.0
        # h[i] * (1 - h[i+k])
        # This is count of 1s in h[0:N-k] where h[i+k] is 0.
        # Or sum of h[0:N-k] - sum of h[0:N-k]*h[k:N]
        # Let's just compute directly for clarity, it's fast enough.
        # Using numpy slicing for speed
        # h_a = h[:N-k]
        # h_b = h[k:]
        # overlap = np.sum(h_a * (1 - h_b))
        
        # Manual loop might be faster for small N in Python if numpy overhead is high?
        # N=100 is small. Numpy is fine.
        overlap = np.sum(h[:N-k] * (1 - h[k:]))
        if overlap > max_val:
            max_val = overlap
    return max_val

def _project_to_binary(h):
    """
    Ensure h is binary (0 or 1) and has exactly N/2 ones.
    """
    h = np.round(h).astype(int)
    # Count ones
    ones = np.sum(h)
    target = N // 2
    if ones > target:
        # Flip some 1s to 0s
        idx_ones = np.where(h == 1)[0]
        # Shuffle and flip
        random.shuffle(idx_ones)
        for i in idx_ones[:ones - target]:
            h[i] = 0
    elif ones < target:
        idx_zeros = np.where(h == 0)[0]
        random.shuffle(idx_zeros)
        for i in idx_zeros[:target - ones]:
            h[i] = 1
    return h

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    Uses local search on binary sequences.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with a random binary sequence
    h = np.zeros(N, dtype=int)
    indices = np.random.permutation(N)
    h[indices[:N//2]] = 1
    
    current_overlap = _max_overlap(h)
    best_h = h.copy()
    best_overlap = current_overlap
    
    # Local Search
    # Swap a 0 and a 1 if it reduces the max overlap.
    # To speed up, we can just try random swaps.
    while time.time() < deadline:
        # Pick a random 1 and a random 0
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        if i == j:
            continue
            
        # Swap
        h[i] = 0
        h[j] = 1
        
        new_overlap = _max_overlap(h)
        
        if new_overlap < best_overlap:
            best_overlap = new_overlap
            best_h = h.copy()
        else:
            # Revert if not better than current best? 
            # Or revert if not better than current state to avoid drift?
            # Standard hill climbing: revert if not better than current.
            # But we want to escape local minima. 
            # Let's try: keep if better than current, revert otherwise.
            if new_overlap >= current_overlap:
                h[i] = 1
                h[j] = 0
            else:
                current_overlap = new_overlap
        
        # Check deadline
        if time.time() > deadline:
            break
            
    return best_h.astype(float)

# EVOLVE-BLOCK-END
