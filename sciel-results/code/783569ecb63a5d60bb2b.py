"""Rewritten program: Local search with simulated annealing to minimize max AP discrepancy."""
import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin from 60s

    # Precompute all arithmetic progressions
    # An AP is (start, step, length) where indices are start, start+step, ..., start+(length-1)*step < n
    # We need to consider all start in [0, n), step in [1, n), and length >= 1
    # For efficiency, we store for each index which APs it belongs to, and for each AP the list of indices
    
    # Actually, for local search, we need to quickly compute the sum of any AP.
    # Let's precompute for each index i, the list of (ap_id, sign_in_ap) pairs.
    # But that's complex. Instead, let's precompute the APs as lists of indices.
    
    aps = []
    # For each step d from 1 to n
    for d in range(1, n):
        # For each start s from 0 to n-1
        for s in range(n):
            idxs = list(range(s, n, d))
            if len(idxs) >= 1:
                aps.append(idxs)
    
    num_aps = len(aps)
    
    # For faster evaluation, let's build a structure:
    # For each index i, store list of (ap_id, position_in_ap) 
    # Actually, we just need the sum. Let's precompute for each AP the indices.
    # When we flip index i, we need to update all APs that contain i.
    # Let's build: index_to_aps[i] = list of ap_ids containing i
    
    index_to_aps = [[] for _ in range(n)]
    for ap_id, idxs in enumerate(aps):
        for idx in idxs:
            index_to_aps[idx].append(ap_id)
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Compute initial AP sums
    ap_sums = [0] * num_aps
    for ap_id, idxs in enumerate(aps):
        s = 0
        for idx in idxs:
            s += signs[idx]
        ap_sums[ap_id] = s
    
    def max_discrepancy():
        return max(abs(s) for s in ap_sums)
    
    best_signs = signs[:]
    best_md = max_discrepancy()
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    current_md = best_md
    
    step = 0
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
        
        # Pick a random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        # Update AP sums
        delta = -2 * old_sign
        for ap_id in index_to_aps[i]:
            ap_sums[ap_id] += delta
        
        new_md = max_discrepancy()
        
        # Accept or reject
        if new_md < current_md:
            current_md = new_md
            if new_md < best_md:
                best_md = new_md
                best_signs = signs[:]
        else:
            # Metropolis criterion
            diff = new_md - current_md
            if diff == 0:
                # Accept with some probability to escape plateaus
                if rng.random() < 0.5:
                    current_md = new_md
                else:
                    # Reject
                    signs[i] = old_sign
                    for ap_id in index_to_aps[i]:
                        ap_sums[ap_id] -= delta
            else:
                # diff > 0
                prob = math.exp(-diff / T)
                if rng.random() < prob:
                    current_md = new_md
                else:
                    # Reject
                    signs[i] = old_sign
                    for ap_id in index_to_aps[i]:
                        ap_sums[ap_id] -= delta
        
        T *= alpha
        step += 1
        
        # Check time
        if time.time() - start_time > time_limit - 2.0:
            break
    
    return best_signs
