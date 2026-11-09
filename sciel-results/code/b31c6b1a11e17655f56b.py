"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute all APs: for each step d and start s, the indices s, s+d, s+2d, ...
    # We need APs of ANY length >= 1, so for each (s, d), we track all prefixes.
    # Actually, the max |sum| over ALL APs is the max over all (s, d, L) of |sum of signs[s:L:d]|.
    # For fixed (s,d), the max prefix sum is achieved at some length. 
    # So for each (s,d), we need to track the running sum as we extend.
    # But for efficiency in SA, we track per-AP sums where each AP is a specific (s,d,L) triple.
    # That's too many. Instead, for each (s,d), the maximum absolute prefix sum is what matters.
    # But when we flip a sign, it affects all APs containing that index.
    
    # Alternative: For each index i, which (s,d) pairs contain i?
    # i is in AP (s,d) if s <= i and (i-s) % d == 0, i.e., s = i - k*d for some k >= 0.
    # So s = i, i-d, i-2d, ... >= 0.
    # For each (s,d), the AP is s, s+d, s+2d, ... up to < n.
    # When we flip index i, for each (s,d) containing i, the AP sum changes by delta.
    # But the "AP sum" we care about is the max absolute prefix sum for that (s,d).
    # This is complex. Let me use a different formulation.
    
    # Simpler: enumerate all APs as (start, step, length) where length >= 1.
    # For each such AP, track its sum. When flipping index i, update all APs containing i.
    # Number of APs: for step d, starts 0..n-1, lengths 1..floor((n-1-start)/d)+1.
    # Total is O(n^2 log n) which for n=256 is manageable (~256^2 * log(256) ≈ 160k * 8 = 1.3M? No)
    # Actually for step d, number of APs = sum over starts of (floor((n-1-start)/d)+1) = sum_{k=1}^{n} ceil(k/d) roughly.
    # For d=1: n(n+1)/2 = 32896
    # For d=2: ~n^2/4 = 16384
    # Total over all d: sum_{d=1}^{n} O(n^2/d) = O(n^2 log n) ≈ 256^2 * 5.5 ≈ 360k. That's fine.
    
    # Let's precompute: for each index i, list of (ap_id, position_in_ap) 
    # Actually we just need the list of ap_ids containing i.
    
    aps = []  # list of index lists
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for s in range(n):
            # AP: s, s+d, s+2d, ... 
            # All prefixes of this sequence are valid APs
            # But we want each distinct AP (s,d,L) as a separate constraint
            # For efficiency, note that for fixed (s,d), the max |prefix sum| is what we need
            # But in SA, tracking each prefix separately is simpler for incremental updates
            # Let's track each (s,d,L) as a separate AP
            idxs = []
            for L in range(1, n):
                end = s + L * d
                if end >= n:
                    break
                idxs.append(end)
            # Now for each prefix length 1..len(idxs), the AP is idxs[:L]
            # That's len(idxs) APs for this (s,d)
            for L in range(1, len(idxs) + 1):
                ap_indices = idxs[:L]
                ap_id = len(aps)
                aps.append(ap_indices)
                for idx in ap_indices:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Initialize with Thue-Morse
    signs = np.array([(-1) ** (bin(i).count('1') % 2) for i in range(n)], dtype=np.int8)
    
    # Compute initial AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for k, idxs in enumerate(aps):
        ap_sums[k] = np.sum(signs[idxs])
    
    # Frequency of absolute sums
    max_possible = n
    freq_abs = np.zeros(max_possible + 1, dtype=np.int32)
    for s in ap_sums:
        freq_abs[abs(s)] += 1
    
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(42)
    
    # Simulated annealing with constraint-based pruning
    # Key: if current_max is B, we can reject any move that would make any AP sum have |sum| > B
    # This is a hard constraint that prunes the search space
    
    T = 3.0
    cooling_rate = 0.99995
    steps = 0
    no_improve_steps = 0
    patience = 5000
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Pick random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        delta = -2 * old_sign
        
        affected_aps = index_to_aps[i]
        
        # Quick check: would this violate the bound?
        # For each affected AP, new_abs = abs(old_sum + delta)
        # If any new_abs > current_max, this move is "bad" (increases max or violates bound)
        # We can still accept it with some probability, but constraint-based approach says:
        # If we're trying to maintain max <= B, reject if any |new_sum| > B
        
        old_vals = ap_sums[affected_aps]
        new_vals = old_vals + delta
        new_abs_vals = np.abs(new_vals)
        
        # Check if any new absolute sum exceeds current_max
        max_new_abs = int(np.max(new_abs_vals)) if len(new_abs_vals) > 0 else 0
        
        # Constraint-based pruning: if max_new_abs > current_max, this move increases the max
        # We can still accept it with low probability (SA), but let's be more aggressive
        # in rejecting moves that significantly increase the max
        
        # Compute cost difference
        # The new global max would be max(current_max, max_new_abs) if we accept
        # But if all APs with |sum| == current_max are affected and decrease, the max might drop
        
        # For simplicity in the constraint approach:
        # Reject if max_new_abs > current_max (hard constraint)
        # Accept with probability if max_new_abs <= current_max
        
        if max_new_abs > current_max:
            # This move would increase the maximum discrepancy
            # Reject with high probability, but allow some exploration
            if T < 0.1:
                accept = False
            else:
                # Probability of accepting a worsening move
                prob = math.exp(-(max_new_abs - current_max) / T)
                accept = rng.random() < prob
        else:
            # This move doesn't increase the max (might decrease it)
            accept = True
        
        if accept:
            # Update
            for idx_ap, old_v, new_v in zip(affected_aps, old_vals, new_vals):
                oa = abs(old_v)
                na = abs(new_v)
                freq_abs[oa] -= 1
                freq_abs[na] += 1
                ap_sums[idx_ap] = new_v
            
            signs[i] = -signs[i]
            
            # Recompute actual max
            # If freq_abs[current_max] == 0, the max has dropped
            actual_new_max = current_max
            if freq_abs[actual_new_max] == 0:
                while actual_new_max > 0 and freq_abs[actual_new_max] == 0:
                    actual_new_max -= 1
            
            current_max = actual_new_max
            
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        # Cool down
        T *= cooling_rate
        
        # If stuck for too long, restart with perturbation
        if no_improve_steps > patience:
            no_improve_steps = 0
            # Perturb: flip a few random signs
            for _ in range(5):
                j = rng.randrange(n)
                signs[j] = -signs[j]
            # Recompute everything
            ap_sums[:] = 0
            for k, idxs in enumerate(aps):
                ap_sums[k] = np.sum(signs[idxs])
            freq_abs[:] = 0
            for s in ap_sums:
                freq_abs[abs(s)] += 1
            current_max = int(np.max(np.abs(ap_sums)))
            T = max(T, 1.0)
    
    # Final local search polish
    if time.time() - start_time < time_limit - 5:
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for pos in range(n):
                old_sign = signs[pos]
                new_sign = -old_sign
                signs[pos] = new_sign
                
                new_max_disc = 0
                for ap_id in index_to_aps[pos]:
                    current_sums_val = ap_sums[ap_id] + 2 * new_sign
                    abs_s = abs(current_sums_val)
                    if abs_s > new_max_disc:
                        new_max_disc = abs_s
                
                if new_max_disc < current_max:
                    # Apply
                    for ap_id in index_to_aps[pos]:
                        ap_sums[ap_id] += 2 * new_sign
                        old_abs = abs(ap_sums[ap_id] - 2 * new_sign)
                        new_abs = abs(ap_sums[ap_id])
                        freq_abs[old_abs] -= 1
                        freq_abs[new_abs] += 1
                    current_max = new_max_disc
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[pos] = old_sign
    
    # Verify and return best
    return list(best_signs)
# EVOLVE-BLOCK-END
