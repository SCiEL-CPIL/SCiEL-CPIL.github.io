import time
import random
import numpy as np
import math

def _build_ap_structure(n):
    """
    Pre-compute the list of APs and their membership for faster local search updates.
    Returns:
        aps: list of lists of indices
        ap_membership: list of lists of ap_indices for each index in [0..n-1]
    """
    aps = []
    ap_membership = [[] for _ in range(n)]
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
                    
    return aps, ap_membership

def _compute_ap_extrema(ap_indices, signs_arr, ap_idx, max_sums, min_sums):
    """
    Compute max and min subarray sums for a specific AP and update arrays in place.
    """
    sub = signs_arr[ap_indices]
    if len(sub) == 0:
        max_sums[ap_idx] = 0
        min_sums[ap_idx] = 0
        return
        
    cum = np.concatenate(([0], np.cumsum(sub)))
    
    # min_cum_before[j] = min(cum[0..j-1])
    min_cum_before = np.empty_like(cum)
    min_cum_before[0] = 0
    if len(cum) > 1:
        min_cum_before[1:] = np.minimum.accumulate(cum[:-1])
        
    # max_cum_before[j] = max(cum[0..j-1])
    max_cum_before = np.empty_like(cum)
    max_cum_before[0] = 0
    if len(cum) > 1:
        max_cum_before[1:] = np.maximum.accumulate(cum[:-1])
        
    # Max subarray sum is max(cum[j] - min_cum_before[j])
    # Min subarray sum is min(cum[j] - max_cum_before[j])
    max_sums[ap_idx] = int(np.max(cum - min_cum_before))
    min_sums[ap_idx] = int(np.min(cum - max_cum_before))

def _get_global_disc(max_sums, min_sums):
    """
    Calculate the global maximum discrepancy from per-AP max/min sums.
    """
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def construct():
    start_time = time.time()
    # 90s limit, leave 5s margin
    deadline = start_time + 85
    
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    n_aps = len(aps)
    
    # Initialization: Thue-Morse
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    # Initialize max_sums and min_sums
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx in range(n_aps):
        _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
        
    current_disc = _get_global_disc(max_sums, min_sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    if best_disc == 1:
        return best_signs

    # Search Loop
    # We alternate between targeted descent on the worst AP and random moves (SA)
    temp = 1.0
    cool_rate = 0.9995
    no_improve_count = 0
    max_no_improve_for_sa = 20 # After this many failed targeted steps, do a random move
    
    while time.time() < deadline:
        # 1. Targeted Descent: Attack the worst AP
        # Find the AP with the highest discrepancy
        discs = np.maximum(np.abs(max_sums), np.abs(min_sums))
        worst_ap_idx = int(np.argmax(discs))
        
        found_improvement = False
        best_flip_idx_in_ap = -1
        best_new_disc = current_disc
        
        # Try flipping each element in the worst AP
        ap_indices = aps[worst_ap_idx]
        # Shuffle to avoid bias
        indices_to_try = ap_indices[:]
        rng.shuffle(indices_to_try)
        
        for idx in indices_to_try:
            old_val = signs_arr[idx]
            signs_arr[idx] = -old_val
            
            # Update all APs containing idx
            for ap_idx in ap_membership[idx]:
                _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
            
            new_disc = _get_global_disc(max_sums, min_sums)
            
            if new_disc < best_new_disc:
                best_new_disc = new_disc
                best_flip_idx_in_ap = idx
                found_improvement = True
                # We don't break because we want to find the *best* flip in this AP
                # But we must revert to test the next one
                # Note: signs_arr is currently flipped for idx
            # Revert flip to test next candidate
            signs_arr[idx] = old_val
            for ap_idx in ap_membership[idx]:
                _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                
        if found_improvement:
            # Apply the best flip found
            idx = best_flip_idx_in_ap
            signs[idx] = -signs[idx]
            signs_arr[idx] = -signs_arr[idx]
            for ap_idx in ap_membership[idx]:
                _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
            
            current_disc = best_new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if best_disc == 1:
                return best_signs
        else:
            # No improvement in the worst AP.
            no_improve_count += 1
            
            # If stuck for a while, do a Simulated Annealing move
            if no_improve_count >= max_no_improve_for_sa:
                # Pick a random index
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                new_val = -old_val
                
                signs[idx] = new_val
                signs_arr[idx] = new_val
                
                for ap_idx in ap_membership[idx]:
                    _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                    
                new_disc = _get_global_disc(max_sums, min_sums)
                
                delta = new_disc - current_disc
                # Accept if better, or with probability exp(-delta/temp)
                if delta < 0 or rng.random() < np.exp(-delta / max(temp, 0.01)):
                    current_disc = new_disc
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert
                    signs[idx] = old_val
                    signs_arr[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                    no_improve_count += 1
                    
                temp *= cool_rate
                if temp < 0.01:
                    temp = 0.01
            else:
                # Try a random flip from the union of top 5 worst APs to escape local minima more gently
                top_5_aps = np.argsort(discs)[-5:]
                candidates = set()
                for ap_idx in top_5_aps:
                    candidates.update(aps[ap_idx])
                if candidates:
                    idx = rng.choice(list(candidates))
                    signs[idx] = -signs[idx]
                    signs_arr[idx] = -signs_arr[idx]
                    for ap_idx in ap_membership[idx]:
                        _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                    new_disc = _get_global_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        if new_disc < best_disc:
                            best_disc = new_disc
                            best_signs = signs[:]
                            no_improve_count = 0
                    else:
                        # Revert if worse? No, this is a hill-climbing step, usually we revert if worse.
                        # But to escape, we might accept. Let's just revert to keep it pure descent for now, 
                        # and rely on the SA step for escape.
                        signs[idx] = -signs[idx]
                        signs_arr[idx] = -signs_arr[idx]
                        for ap_idx in ap_membership[idx]:
                            _compute_ap_extrema(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                        no_improve_count += 1

    return best_signs
