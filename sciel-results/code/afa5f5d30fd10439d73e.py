"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """Compute maximum absolute signed sum over all arithmetic progressions in [1, n]."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons have abs sum 1
    
    # Iterate over all possible steps
    for step in range(1, n + 1):
        # For a given step, the arithmetic progressions are determined by their start index r.
        # The valid starts are r such that r < n.
        # The sequence for start r is arr[r], arr[r+step], arr[r+2*step], ...
        # These sequences for different r (0 <= r < step) are disjoint and cover all elements.
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            # Compute prefix sums to find max absolute sum over all contiguous subarrays
            # Wait, arithmetic progression sum is the sum of ALL elements in that progression.
            # The problem says "maximum absolute signed sum on ANY arithmetic progression".
            # An arithmetic progression is defined by start, step, and length.
            # So for a fixed start r and step s, the progression of length L is arr[r], arr[r+s], ..., arr[r+(L-1)*s].
            # We need to check ALL possible lengths L for each (r, s).
            # This means for each (r, s), we look at the sequence subseq = arr[r::s].
            # We need the maximum absolute value of any prefix sum of subseq?
            # No, the sum is of the progression itself. A progression is a contiguous segment of the infinite sequence?
            # "contained in [1,256]".
            # An arithmetic progression is a set of numbers {a, a+d, a+2d, ..., a+kd}.
            # The sum is sum_{i=0}^k signs[a+id] (adjusting for 0-indexing).
            # So for a fixed (start, step), the possible progressions are prefixes of the sequence arr[start::step].
            # So we need the max absolute prefix sum for each (start, step).
            
            # Let's verify.
            # If the progression is just one element, sum is signs[start].
            # If length 2, signs[start] + signs[start+step].
            # So yes, for each (start, step), we look at the sequence S = arr[start::step].
            # We compute prefix sums P[0]=S[0], P[1]=S[0]+S[1], ..., P[k]=sum(S[0..k]).
            # The max absolute sum for this (start, step) is max(abs(P[i]) for i in range(len(S))).
            # And we take the max over all (start, step).
            
            # We can vectorize the prefix sum computation.
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
    
    return max_disc

def _local_search(signs, max_iterations=5000, time_limit=55.0):
    """Perform local search to minimize maximum absolute signed sum."""
    n = len(signs)
    signs = list(signs)
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(42)
    
    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly pick a position to flip
        idx = rng.randint(0, n - 1)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_disc = _compute_max_discrepancy(signs)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = signs.copy()
            current_disc = new_disc
        else:
            # Revert if not better (or equal, to avoid random walk)
            signs[idx] = old_sign
    
    return best_signs, best_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s
    
    # Initial random start
    rng = random.Random(171)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    # Run local search
    best_signs, best_disc = _local_search(signs, max_iterations=10000, time_limit=deadline - (time.time() - start_time))
    
    # Check time, if still have time, try more
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try a few more restarts if time permits
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
            signs = [rng.choice((-1, 1)) for _ in range(n)]
            new_signs, new_disc = _local_search(signs, max_iterations=10000, time_limit=deadline - (time.time() - start_time))
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = new_signs
    
    return best_signs
# EVOLVE-BLOCK-END
