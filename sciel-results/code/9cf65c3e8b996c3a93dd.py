"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    # Precompute all arithmetic progressions
    # An AP is defined by (start, step) where indices are start, start+step, start+2*step, ...
    # We need all APs with at least 1 element (singletons included)
    # For efficiency, we'll track which APs each index belongs to
    
    # Build list of APs as tuples of indices
    aps = []
    # index_to_aps[i] = list of AP indices that contain index i
    index_to_aps = [[] for _ in range(n)]
    
    # Singletons: each index is its own AP of length 1
    for i in range(n):
        ap_id = len(aps)
        aps.append((i,))
        index_to_aps[i].append(ap_id)
    
    # Multi-element APs
    for step in range(1, n):
        for start in range(n):
            indices = []
            idx = start
            while idx < n:
                indices.append(idx)
                idx += step
            if len(indices) >= 2:
                ap_id = len(aps)
                aps.append(tuple(indices))
                for idx in indices:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Precompute AP sums as arrays for fast evaluation
    # ap_sums[j] = sum of signs over AP j
    # We'll maintain these incrementally
    
    def compute_ap_sums(signs):
        """Compute all AP sums from scratch."""
        sums = [0] * num_aps
        for j, ap in enumerate(aps):
            s = 0
            for idx in ap:
                s += signs[idx]
            sums[j] = s
        return sums
    
    def max_abs_sum(ap_sums):
        """Find max absolute AP sum."""
        m = 0
        for s in ap_sums:
            a = -s if s < 0 else s
            if a > m:
                m = a
        return m
    
    # Initialize with random signs
    rng = random.Random(12345)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    ap_sums = compute_ap_sums(signs)
    best_max = max_abs_sum(ap_sums)
    
    # Greedy improvement: flip signs that reduce max discrepancy
    def greedy_improve(signs, ap_sums, max_iter=100000):
        """Iteratively flip signs to reduce max absolute AP sum."""
        best_max = max_abs_sum(ap_sums)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() - start_time > time_limit - 5:
                break
            for i in range(n):
                # Try flipping sign at position i
                delta = 2 * signs[i]  # change in sum when flipping
                old_signs_i = signs[i]
                signs[i] = -signs[i]
                
                # Update affected AP sums
                for ap_id in index_to_aps[i]:
                    ap_sums[ap_id] -= delta
                
                new_max = max_abs_sum(ap_sums)
                
                if new_max < best_max:
                    best_max = new_max
                    improved = True
                else:
                    # Revert
                    signs[i] = old_signs_i
                    for ap_id in index_to_aps[i]:
                        ap_sums[ap_id] += delta
        return signs, ap_sums, best_max
    
    signs, ap_sums, best_max = greedy_improve(signs, ap_sums)
    
    # Simulated annealing with multi-flip moves
    def simulated_annealing(signs, ap_sums, best_max, iters_per_temp=5000):
        """Simulated annealing with single and multi-flip moves."""
        current_max = best_max
        T = 5.0
        cooling = 0.995
        best_signs = list(signs)
        best_ap_sums = list(ap_sums)
        best_max_found = current_max
        
        while T > 0.1 and time.time() - start_time < time_limit - 3:
            for _ in range(iters_per_temp):
                if time.time() - start_time > time_limit - 3:
                    break
                
                # Choose move type: single flip or multi-flip
                if rng.random() < 0.7:
                    # Single flip
                    i = rng.randrange(n)
                    flipped = [i]
                else:
                    # Multi-flip: flip 2-4 random positions
                    k = rng.randint(2, 4)
                    flipped = rng.sample(range(n), k)
                
                # Apply flip
                for i in flipped:
                    delta = 2 * signs[i]
                    signs[i] = -signs[i]
                    for ap_id in index_to_aps[i]:
                        ap_sums[ap_id] -= delta
                
                new_max = max_abs_sum(ap_sums)
                
                if new_max <= current_max:
                    current_max = new_max
                    if new_max < best_max_found:
                        best_max_found = new_max
                        best_signs = list(signs)
                        best_ap_sums = list(ap_sums)
                else:
                    # Accept with probability exp(-(new_max - current_max)/T)
                    delta = new_max - current_max
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        current_max = new_max
                    else:
                        # Revert
                        for i in flipped:
                            delta = 2 * signs[i]
                            signs[i] = -signs[i]
                            for ap_id in index_to_aps[i]:
                                ap_sums[ap_id] += delta
                
                T *= cooling
        
        return best_signs, best_ap_sums, best_max_found
    
    signs, ap_sums, best_max = simulated_annealing(signs, ap_sums, best_max)
    
    # Final greedy polish
    signs, ap_sums, best_max = greedy_improve(signs, ap_sums, max_iter=50000)
    
    return signs

# EVOLVE-BLOCK-END
