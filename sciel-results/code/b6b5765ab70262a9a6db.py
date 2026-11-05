"""Low discrepancy on all arithmetic progressions - local search approach."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs, limit=None):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1.
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
    
    # Initialize with random signs
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_val = _max_ap_discrepancy(best_signs)
    
    # Steepest Descent Local Search with Random Restarts
    while time.time() < deadline - 5:
        current_signs = best_signs[:]
        current_val = best_val
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_flip = -1
            best_flip_val = current_val
            
            for i in range(n):
                current_signs[i] = -current_signs[i]
                # Use limit to speed up evaluation of non-improving moves
                v = _max_ap_discrepancy(current_signs, limit=best_flip_val)
                if v < best_flip_val:
                    best_flip_val = v
                    best_flip = i
                current_signs[i] = -current_signs[i] # revert
            
            if best_flip != -1:
                current_signs[best_flip] = -current_signs[best_flip]
                current_val = best_flip_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_signs = current_signs[:]
        
        # Random restart
        if time.time() < deadline - 10:
             signs = [rng.choice((-1, 1)) for _ in range(n)]
             val = _max_ap_discrepancy(signs)
             if val < best_val:
                 best_signs = signs[:]
                 best_val = val
    
    return best_signs
# EVOLVE-BLOCK-END
