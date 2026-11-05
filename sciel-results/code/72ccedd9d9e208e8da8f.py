"""Low discrepancy on all arithmetic progressions - 3-Flip Local Search with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def _get_max_ap_discrepancy_vectorized(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns the maximum absolute discrepancy.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    
    max_abs = 0
    
    for d in range(1, n):
        max_start = n - d
        if max_start <= 0:
            continue
        
        for i in range(max_start):
            seq = s[i:n:d]
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                
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
    
    # 3-Flip Local Search
    # We explore neighborhoods of size 3 (triple flips).
    # This allows escaping local minima where single or double flips are blocked.
    # To manage combinatorial explosion (256 choose 3 ~ 2.7M), we use a stochastic sampling approach
    # combined with a greedy repair loop.
    
    # Strategy:
    # 1. Find the worst AP.
    # 2. Try to fix it by flipping a subset of its elements.
    # 3. Since flipping 3 elements in a long AP is complex to optimize directly, 
    #    we use a hybrid: 
    #    - Identify the worst AP.
    #    - Generate candidate 3-flip moves that involve at least one element from the worst AP 
    #      (to directly address the bottleneck) and two random other elements (to escape local minima).
    #    - Evaluate and accept if improvement.
    #    - If no improvement found in a batch, fall back to random 3-flips.
    
    # Pre-compute indices for speed? No, just use numpy operations.
    
    # Helper to get the worst AP (start, step, length)
    def _get_worst_ap(signs):
        n = len(signs)
        s = np.asarray(signs, dtype=np.int32)
        max_abs = 0
        best_i = 0
        best_d = 1
        best_len = 1
        
        for d in range(1, n):
            max_start = n - d
            if max_start <= 0:
                continue
            for i in range(max_start):
                seq = s[i:n:d]
                cumsum = np.cumsum(seq)
                abs_cumsum = np.abs(cumsum)
                current_max = np.max(abs_cumsum)
                if current_max > max_abs:
                    max_abs = current_max
                    best_i = i
                    best_d = d
                    # Find the length that achieved this max
                    # The max could be at any prefix length. We just need the indices of the AP.
                    # The AP is i, i+d, i+2d...
                    # We don't strictly need the exact length that caused the max for the flip selection,
                    # just the indices of the AP elements.
                    
        return best_i, best_d, max_abs

    # Main loop
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        # Get worst AP
        worst_i, worst_d, worst_val = _get_worst_ap(signs)
        
        # Get indices of the worst AP
        ap_indices = list(range(worst_i, n, worst_d))
        
        # Generate candidate 3-flip moves
        # We want to flip 3 indices.
        # Heuristic: Include at least one index from the worst AP to try to reduce its sum.
        # The other two can be random to explore the space.
        
        found_improvement = False
        
        # Try a few candidates
        num_candidates = 50
        for _ in range(num_candidates):
            # Pick 1 from AP, 2 from anywhere (distinct)
            idx1 = rng.choice(ap_indices)
            # Pick 2 distinct indices from 0..n-1, not equal to idx1
            # To ensure distinctness efficiently:
            other_indices = rng.choice(n, size=2, replace=False)
            # Check if they collide with idx1 or each other
            if other_indices[0] == idx1 or other_indices[1] == idx1 or other_indices[0] == other_indices[1]:
                continue
            
            idx2, idx3 = other_indices
            
            # Perform flip
            signs[idx1] = -signs[idx1]
            signs[idx2] = -signs[idx2]
            signs[idx3] = -signs[idx3]
            
            new_disc = _get_max_ap_discrepancy_vectorized(signs)
            
            if new_disc < current_disc:
                current_disc = new_disc
                found_improvement = True
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs.copy()
                    if best_disc == 1:
                        break
                no_improve_count = 0
                break
            else:
                # Revert
                signs[idx1] = -signs[idx1]
                signs[idx2] = -signs[idx2]
                signs[idx3] = -signs[idx3]
        
        if not found_improvement:
            no_improve_count += 1
            # If stuck for too long, try a random 3-flip without bias to worst AP
            if no_improve_count % 100 == 0:
                indices = rng.choice(n, size=3, replace=False)
                signs[indices[0]] = -signs[indices[0]]
                signs[indices[1]] = -signs[indices[1]]
                signs[indices[2]] = -signs[indices[2]]
                new_disc = _get_max_ap_discrepancy_vectorized(signs)
                if new_disc < current_disc:
                    current_disc = new_disc
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs.copy()
                        if best_disc == 1:
                            break
                    no_improve_count = 0
                else:
                    signs[indices[0]] = -signs[indices[0]]
                    signs[indices[1]] = -signs[indices[1]]
                    signs[indices[2]] = -signs[indices[2]]
            
            # If stuck for a very long time, restart with a new random configuration?
            # Or just keep trying. 3-flips provide more escape routes than 1-flips.
            if no_improve_count > max_no_improve:
                # Restart
                signs = rng.choice([-1, 1], size=n).astype(np.int32)
                current_disc = _get_max_ap_discrepancy_vectorized(signs)
                no_improve_count = 0
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs.copy()

    return best_signs.tolist()

# EVOLVE-BLOCK-END
