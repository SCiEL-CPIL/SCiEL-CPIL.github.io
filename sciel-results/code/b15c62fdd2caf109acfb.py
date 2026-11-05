import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50 # 50s margin

    # 1. Precompute APs
    # ap_indices[k] is a list of indices for the k-th AP
    # We iterate d from 1 to n-1, and r from 0 to d-1
    # AP is r, r+d, r+2d, ... < n
    ap_indices = []
    ap_containing = [[] for _ in range(n)]
    
    ap_id = 0
    for d in range(1, n):
        for r in range(d):
            indices = []
            idx = r
            while idx < n:
                indices.append(idx)
                idx += d
            if len(indices) > 0:
                ap_indices.append(indices)
                for idx in indices:
                    ap_containing[idx].append(ap_id)
                ap_id += 1
                
    num_aps = len(ap_indices)
    
    # Helper to calculate sums from scratch
    def calc_sums(signs):
        sums = [0] * num_aps
        for k, indices in enumerate(ap_indices):
            s = 0
            for idx in indices:
                s += signs[idx]
            sums[k] = s
        return sums

    # Helper to get max abs sum and index of worst AP
    def get_max_info(sums):
        max_val = 0
        max_idx = 0
        for k, s in enumerate(sums):
            abs_s = abs(s)
            if abs_s > max_val:
                max_val = abs_s
                max_idx = k
        return max_val, max_idx

    # Initialize
    # Use a heuristic initialization: Thue-Morse sequence has low discrepancy for many structures
    # t_i = (-1)^(popcount(i))
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    ap_sums = calc_sums(signs)
    current_max, worst_ap_idx = get_max_info(ap_sums)
    
    best_max = current_max
    best_signs = signs[:]
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 100
    perturb_size = 5
    
    while time.time() < deadline:
        # 1. Targeted Descent
        # Find the AP with max abs sum
        # We maintain current_max and worst_ap_idx approximately, 
        # but to be safe and simple, let's find it exactly each time or maintain it.
        # Maintaining it is faster.
        
        # Let's try to improve the worst AP
        worst_indices = ap_indices[worst_ap_idx]
        worst_sum = ap_sums[worst_ap_idx]
        
        best_flip_idx = -1
        best_new_global_max = current_max
        
        # Try flipping each element in the worst AP
        for idx in worst_indices:
            # Simulate flip
            old_sign = signs[idx]
            new_sign = -old_sign
            
            # Update sums for all APs containing idx
            # We need to check if this flip reduces the global max
            # To do this efficiently, we need to know the new sums.
            # Since we only change sums for APs in ap_containing[idx], 
            # we can track the max among those and the rest.
            
            # This is getting complex to do without a full recalc or a very careful data structure.
            # Given N=256, a full recalc of the max (iterating 32k APs) is ~32k ops.
            # Doing this for each element in the worst AP (len ~ 10-50) is 32k * 50 = 1.6M ops.
            # This might be too slow for 50 seconds if we do it many times.
            
            # Alternative: Just flip one element and check.
            # Greedy: Flip the element that most reduces the worst AP's sum.
            # If the worst AP has sum S > 0, we want to flip a +1 to -1.
            # If S < 0, we want to flip a -1 to +1.
            
            candidates = []
            if worst_sum > 0:
                for idx in worst_indices:
                    if signs[idx] == 1:
                        candidates.append(idx)
            else:
                for idx in worst_indices:
                    if signs[idx] == -1:
                        candidates.append(idx)
                        
            if not candidates:
                break
                
            # Pick a random candidate from the best ones? Or just the first?
            # Let's pick the one that appears in the most other "bad" APs?
            # Too complex. Just pick one randomly from candidates to add diversity.
            idx_to_flip = random.choice(candidates)
            
            # Apply flip
            signs[idx_to_flip] *= -1
            delta = 2 * signs[idx_to_flip] # -2 if flipped to -1, +2 if flipped to 1
            
            # Update ap_sums
            for ap_idx in ap_containing[idx_to_flip]:
                ap_sums[ap_idx] += delta
                
            # Recalculate global max
            new_max, new_worst_idx = get_max_info(ap_sums)
            
            if new_max < current_max:
                current_max = new_max
                worst_ap_idx = new_worst_idx
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs[:]
            elif new_max == current_max:
                # Accept if it changes the configuration (to escape symmetry)
                # But keep worst_ap_idx updated
                worst_ap_idx = new_worst_idx
                # no_improve_count += 1 # Maybe don't increment if we made a move
            else:
                # Revert
                signs[idx_to_flip] *= -1
                for ap_idx in ap_containing[idx_to_flip]:
                    ap_sums[ap_idx] -= delta
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Perturb
            for _ in range(perturb_size):
                idx = random.randint(0, n-1)
                signs[idx] *= -1
                delta = 2 * signs[idx]
                for ap_idx in ap_containing[idx]:
                    ap_sums[ap_idx] += delta
            
            current_max, worst_ap_idx = get_max_info(ap_sums)
            no_improve_count = 0

    return best_signs
