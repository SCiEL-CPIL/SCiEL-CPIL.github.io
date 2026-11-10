import time
import random
import numpy as np

def _max_abs_ap_sum(signs, threshold=None):
    """
    Compute the maximum absolute sum over any arithmetic progression
    contained in [1, n].
    signs: list or numpy array of length n, values -1 or 1.
    threshold: if provided, early-exit and return threshold+1 if exceeded.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            idx = np.arange(s, n, d)
            if len(idx) == 0:
                continue
            chain = signs_arr[idx]
            prefix = np.concatenate(([0], np.cumsum(chain)))
            current_max = int(prefix.max() - prefix.min())
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    
    return max_sum

def _eval_flip(signs, idx, current_max):
    """
    Evaluate the effect of flipping signs[idx].
    Returns the new maximum absolute AP sum.
    Uses incremental update for speed.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8).copy()
    signs_arr[idx] = -signs_arr[idx]
    
    # We need to recompute max abs AP sum, but we can early-exit
    # if we exceed current_max
    return _max_abs_ap_sum(signs_arr, threshold=current_max)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 256
    rng = random.Random(42)
    
    # Initial guess: use a structured pattern then random
    # Start with alternating pattern
    signs = [1 if i % 2 == 0 else -1 for i in range(n)]
    best_signs = signs[:]
    best_score = _max_abs_ap_sum(best_signs)
    
    # Local Search
    current_signs = best_signs[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 100  # Restart if stuck
    
    # Pre-compute for each index, which (d, s) pairs include it
    # This allows faster evaluation of the impact of a flip
    # But for now, use the early-exit evaluator which is already fast
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with random
            current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            continue
        
        # Try flipping a random index
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        # Use early-exit evaluation with current_score as threshold
        new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_signs = current_signs[:]
        else:
            # Revert
            current_signs[idx] = old_val
            no_improve_count += 1
            
    return best_signs
