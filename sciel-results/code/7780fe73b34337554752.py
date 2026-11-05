"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _get_max_ap_discrepancy_vectorized(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Uses vectorization for each step size to speed up evaluation.
    Returns the maximum absolute discrepancy.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    
    max_abs = 0
    
    # Iterate over possible steps
    # Step d can be at most n-1, but practically steps > n/2 only have length 1 or 2 APs.
    # Length 1 APs always have sum +/-1.
    # We care about steps where length >= 2.
    for d in range(1, n):
        # For step d, the starts are 0 to d-1.
        # However, if start + d >= n, the length is 1.
        # We only need to check starts i where i + d < n, i.e., i < n - d.
        max_start = n - d
        
        if max_start <= 0:
            continue
            
        # We need to compute prefix sums for sequences s[i], s[i+d], s[i+2d]...
        # This is tricky to fully vectorize across all i and d at once without huge memory.
        # But we can vectorize over 'i' for a fixed 'd'.
        
        # Create indices for the first 'max_start' elements of the grid for step d?
        # Actually, let's just loop over starts for this d, but vectorize the cumsum?
        # No, cumsum is per sequence.
        
        # Alternative: For fixed d, consider the array of all elements.
        # The APs are columns of a matrix if we reshape? No.
        
        # Let's stick to a hybrid: Loop d, then loop i, but use numpy for the inner cumsum.
        # The current seed does this. Is it fast enough?
        # n=256. Sum of n/d for d=1..n is n log n ~ 256 * 5 ~ 1280 iterations.
        # Each iteration does a cumsum of length <= 256.
        # This is very fast in NumPy. The bottleneck in the seed might be the Python loop overhead 
        # or the fact that it updates best_start/step/len which adds overhead.
        # We just need the value.
        
        for i in range(max_start):
            seq = s[i:n:d]
            # Length of seq is at least 2 because i < n-d
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                # Early exit if we hit a theoretical bound? 
                # For n=256, max possible is 256.
                # If we reach a high number, we can't easily exit early unless we know a lower bound.
                
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe (90s kill, 5s margin)
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int32)
    
    # Evaluate initial
    current_disc = _get_max_ap_discrepancy_vectorized(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will perform multiple passes or a long loop
    # To make SA effective, we need to accept moves that increase discrepancy.
    # We need to re-evaluate the full discrepancy after each move to know the delta.
    # This is expensive (O(n log n) per move).
    # With 55s budget, how many moves can we do?
    # If eval takes ~1ms, we can do ~50,000 moves.
    # If eval takes ~0.1ms, we can do ~500,000 moves.
    
    # Optimization: 
    # Flipping a sign at index k affects all APs containing k.
    # It is hard to calculate delta_disc efficiently without full re-eval or complex bookkeeping.
    # Given n=256 is small, full re-eval is likely fast enough in NumPy.
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        # Flip
        signs[idx] = -signs[idx]
        
        # Evaluate new discrepancy
        new_disc = _get_max_ap_discrepancy_vectorized(signs)
        
        # Calculate delta
        delta = new_disc - current_disc
        
        # Acceptance criterion
        if delta < 0:
            # Improvement: always accept
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
                # If we reach 1, we can stop (theoretical min for n>1 with +/-1 signs? 
                # Actually, sum of all signs is even. Max AP sum can be 1 if balanced?
                # If we get 1, it's optimal.
                if best_disc == 1:
                    break
        else:
            # Worse or equal: accept with probability exp(-delta / T)
            if delta == 0:
                # Equal discrepancy: accept to explore
                current_disc = new_disc
            else:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_disc = new_disc
                else:
                    # Reject: revert flip
                    signs[idx] = -signs[idx]
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            # Restart with higher temperature if stuck?
            # Or just keep going at T_min.
            # Let's do a simple restart: if no improvement in last X steps, reset T?
            # For simplicity, just let it run. The cooling is slow.
            T = T_min

    return best_signs.tolist()

# EVOLVE-BLOCK-END
