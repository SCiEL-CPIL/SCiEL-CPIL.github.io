"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _fast_max_abs_sum(signs_np, n):
    """
    Compute the maximum absolute signed sum over any arithmetic progression
    contained in [1, n].
    signs_np: 1D numpy array of length n+1, index 0 unused, values ±1.
    Returns the max absolute sum.
    """
    best = 0
    # Iterate over all possible steps d
    for d in range(1, n + 1):
        # For a fixed step d, we need to check all starts s such that s+d-1 <= n?
        # Actually, an AP is defined by start s, step d, length L.
        # The sum is signs[s] + signs[s+d] + ... + signs[s+(L-1)d].
        # We want the max abs prefix sum for each sequence starting at s.
        
        # Valid starts s are 1 to n.
        # For each s, the sequence is signs[s], signs[s+d], ...
        
        # Optimization: 
        # For each d, we can compute the prefix sums for all starts simultaneously?
        # Let's just loop over starts but use numpy slicing for the sequence if possible?
        # Slicing signs_np[s::d] gives the sequence.
        # Then np.cumsum gives prefix sums.
        # Then max(abs(cumsum)) gives the max abs sum for that start.
        
        # We need to do this for all s in 1..n.
        # This is O(n) numpy operations per d, each of size ~n/d.
        # Total work ~ sum_d n * (n/d) ~ n^2 log n.
        # With n=256, this is fast.
        
        for s in range(1, n + 1):
            # Extract the sequence
            seq = signs_np[s::d]
            if len(seq) == 0:
                continue
            # Compute prefix sums
            prefix_sums = np.cumsum(seq)
            # Get max absolute value
            max_abs = np.max(np.abs(prefix_sums))
            if max_abs > best:
                best = max_abs
                # Early exit if we find a very large value? 
                # The theoretical max is n. If we hit n, we can stop?
                # But we want to minimize, so we just need the current max to evaluate moves.
                # We can't early exit the whole function unless best == n, 
                # but even then, it's just 256.
                
    return int(best)

def construct():
    start_time = time.time()
    # Hard limit is 60s. Use 55s to be safe.
    deadline = start_time + 55.0
    
    n = 256
    
    # Use numpy for faster evaluation
    # signs_np[0] is unused, signs_np[1..n] are the values
    signs_np = np.zeros(n + 1, dtype=np.int8)
    
    rng = random.Random(42)
    
    # Initialize randomly
    for i in range(1, n + 1):
        signs_np[i] = rng.choice([-1, 1])
    
    current_best = _fast_max_abs_sum(signs_np, n)
    best_signs_np = signs_np.copy()
    
    # Local Search: Simulated Annealing
    # We flip one position at a time.
    # To speed up, we can just flip, evaluate, and decide.
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randint(1, n)
        signs_np[i] *= -1
        
        new_best = _fast_max_abs_sum(signs_np, n)
        delta = new_best - current_best
        
        if delta <= 0:
            current_best = new_best
            if new_best < _fast_max_abs_sum(best_signs_np, n):
                # This check is expensive, so we should track best_score separately
                pass
        
        # Better bookkeeping
        # Let's rewrite the loop with proper bookkeeping
        
    # The above loop was pseudocode. Let's implement it properly.
    
    # Reset for actual run
    signs_np[:] = 0
    for i in range(1, n + 1):
        signs_np[i] = rng.choice([-1, 1])
    
    current_score = _fast_max_abs_sum(signs_np, n)
    best_score = current_score
    best_signs_np = signs_np.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        i = rng.randint(1, n)
        signs_np[i] *= -1
        new_score = _fast_max_abs_sum(signs_np, n)
        delta = new_score - current_score
        
        if delta <= 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs_np = signs_np.copy()
        else:
            # Accept with prob exp(-delta/T)
            if T > T_min and rng.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Reject: revert
                signs_np[i] *= -1
        
        T *= cooling
        
        if T < T_min:
            # Reheat
            T = 1.0
            # Optionally restart from best
            if rng.random() < 0.2:
                signs_np[:] = best_signs_np
                current_score = best_score
    
    return best_signs_np[1:].tolist()
# EVOLVE-BLOCK-END
