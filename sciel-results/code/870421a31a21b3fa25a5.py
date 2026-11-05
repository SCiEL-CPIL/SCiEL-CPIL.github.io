import time
import random
import numpy as np
from collections import defaultdict

def _build_ap_structure(n):
    """Precompute all arithmetic progressions and their indices."""
    # For each position i, store list of (ap_id, sign) where sign is +1 or -1
    # based on whether i contributes positively or negatively to the AP sum
    # Actually, for discrepancy we need max subarray sum over each AP
    # So we need to track prefix sums per AP
    
    # Better approach: for each AP, store the list of indices
    # Then for incremental updates, when we flip index i, we update all APs containing i
    
    # Let's precompute for each index i, which APs contain it
    # And for each AP, the ordered list of indices
    
    ap_indices = []  # list of lists of indices for each AP
    index_to_aps = defaultdict(list)  # index i -> list of (ap_id, position_in_ap)
    
    ap_id = 0
    for d in range(1, n + 1):
        for r in range(d):
            indices = list(range(r, n, d))
            if len(indices) >= 1:
                for pos, idx in enumerate(indices):
                    index_to_aps[idx].append((ap_id, pos))
                ap_indices.append(indices)
                ap_id += 1
    
    return ap_indices, index_to_aps, ap_id

def _compute_initial_ap_extrema(signs, ap_indices):
    """Compute max subarray sum (both positive and negative) for each AP."""
    max_pos = []  # max subarray sum for each AP
    max_neg = []  # min subarray sum (most negative) for each AP
    
    for indices in ap_indices:
        # Compute prefix sums
        cur = 0
        min_pref = 0
        max_pref = 0
        max_sub = 0
        min_sub = 0
        
        for idx in indices:
            cur += signs[idx]
            
            diff = cur - min_pref
            if diff > max_sub:
                max_sub = diff
            if cur < min_pref:
                min_pref = cur
            
            diff_min = cur - max_pref
            if diff_min < min_sub:
                min_sub = diff_min
            if cur > max_pref:
                max_pref = cur
        
        max_pos.append(max_sub)
        max_neg.append(min_sub)
    
    return max_pos, max_neg

def _get_max_disc(max_pos, max_neg):
    """Get current maximum absolute discrepancy."""
    if not max_pos:
        return 0
    return max(max(max_pos), -min(max_neg))

def _update_ap_extrema(signs, ap_indices, ap_id, pos_in_ap, max_pos, max_neg):
    """Update the extrema for a single AP after a flip.
    This is expensive if we recompute from scratch, but necessary for correctness."""
    indices = ap_indices[ap_id]
    
    cur = 0
    min_pref = 0
    max_pref = 0
    max_sub = 0
    min_sub = 0
    
    for idx in indices:
        cur += signs[idx]
        
        diff = cur - min_pref
        if diff > max_sub:
            max_sub = diff
        if cur < min_pref:
            min_pref = cur
        
        diff_min = cur - max_pref
        if diff_min < min_sub:
            min_sub = diff_min
        if cur > max_pref:
            max_pref = cur
    
    max_pos[ap_id] = max_sub
    max_neg[ap_id] = min_sub

def _max_ap_discrepancy(signs, limit=None):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    limit: if provided, returns limit as soon as discrepancy >= limit is found.
    Returns the max absolute sum.
    """
    n = len(signs)
    max_abs = 0
    
    for d in range(1, n + 1):
        for r in range(d):
            min_prefix = 0
            max_prefix = 0
            cur_sum = 0
            max_sub = 0
            min_sub = 0
            
            idx = r
            while idx < n:
                cur_sum += signs[idx]
                
                diff = cur_sum - min_prefix
                if diff > max_sub:
                    max_sub = diff
                    if limit is not None and max_sub >= limit:
                        return limit
                if cur_sum < min_prefix:
                    min_prefix = cur_sum
                    
                diff_min = cur_sum - max_prefix
                if diff_min < min_sub:
                    min_sub = diff_min
                    if limit is not None and -min_sub >= limit:
                        return limit
                if cur_sum > max_prefix:
                    max_prefix = cur_sum
                
                idx += d
                
            ap_max = max(max_sub, -min_sub)
            if ap_max > max_abs:
                max_abs = ap_max
                if limit is not None and max_abs >= limit:
                    return limit
                    
    return max_abs

def construct():
    deadline = time.time() + 50  # 10s safety margin
    n = 256
    rng = random.Random(42)
    
    # Precompute AP structure
    ap_indices, index_to_aps, total_aps = _build_ap_structure(n)
    
    # Initialize with random signs
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_max_pos, best_max_neg = _compute_initial_ap_extrema(best_signs, ap_indices)
    best_val = _get_max_disc(best_max_pos, best_max_neg)
    
    # Steepest Descent with incremental updates
    while time.time() < deadline - 5:
        current_signs = best_signs[:]
        current_max_pos = best_max_pos[:]
        current_max_neg = best_max_neg[:]
        current_val = best_val
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            best_flip = -1
            best_flip_val = current_val
            
            for i in range(n):
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                # Update all APs that contain index i
                for ap_id, pos in index_to_aps[i]:
                    _update_ap_extrema(current_signs, ap_indices, ap_id, pos, current_max_pos, current_max_neg)
                
                v = _get_max_disc(current_max_pos, current_max_neg)
                if v < best_flip_val:
                    best_flip_val = v
                    best_flip = i
                    if v < 1:  # Can't do better than 1
                        current_signs[i] = old_sign  # revert for now, will commit below
                        break
                else:
                    current_signs[i] = old_sign  # revert
                
            if best_flip != -1:
                # Commit the best flip
                current_signs[best_flip] = -current_signs[best_flip]
                for ap_id, pos in index_to_aps[best_flip]:
                    _update_ap_extrema(current_signs, ap_indices, ap_id, pos, current_max_pos, current_max_neg)
                current_val = best_flip_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_signs = current_signs[:]
                    best_max_pos = current_max_pos[:]
                    best_max_neg = current_max_neg[:]
        
        # Random restart with perturbation
        if time.time() < deadline - 10:
            signs = best_signs[:]
            # Perturb 1-3 random positions
            num_perturb = rng.randint(1, 3)
            for _ in range(num_perturb):
                i = rng.randint(0, n - 1)
                signs[i] = -signs[i]
            
            max_pos, max_neg = _compute_initial_ap_extrema(signs, ap_indices)
            val = _get_max_disc(max_pos, max_neg)
            if val < best_val:
                best_signs = signs[:]
                best_val = val
                best_max_pos = max_pos
                best_max_neg = max_neg
    
    return best_signs
