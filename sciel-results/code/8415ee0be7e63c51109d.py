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
            if s >= n:
                continue
            # Compute prefix sums incrementally
            p = 0
            min_p = 0
            max_p = 0
            idx = s
            while idx < n:
                p += signs_arr[idx]
                if p < min_p:
                    min_p = p
                elif p > max_p:
                    max_p = p
                idx += d
            current_max = max_p - min_p
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    
    return max_sum

def _get_tight_ap_info(signs, top_k=40):
    """
    Find indices that appear in APs with sums close to the maximum.
    Returns a list of candidate indices to flip, weighted by tightness.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    ap_sums = []
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            p = 0
            min_p = 0
            max_p = 0
            idx = s
            while idx < n:
                p += signs_arr[idx]
                if p < min_p:
                    min_p = p
                elif p > max_p:
                    max_p = p
                idx += d
            current_max = max_p - min_p
            if current_max > 0:
                ap_sums.append((current_max, s, d))
            if current_max > max_sum:
                max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    tight_threshold = max_sum - 1
    tight_aps = [(s, d) for (val, s, d) in ap_sums if val >= tight_threshold]
    
    idx_count = np.zeros(n, dtype=np.int32)
    for s, d in tight_aps:
        idx = s
        while idx < n:
            idx_count[idx] += 1
            idx += d
    
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    rng = random.Random(42)
    
    # Multiple initializations
    initializations = []
    
    # 1. Alternating
    initializations.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # 2. Thue-Morse
    tm = []
    for i in range(n):
        tm.append(1 if bin(i).count('1') % 2 == 0 else -1)
    initializations.append(tm)
    
    # 3-6. Random starts
    for _ in range(4):
        initializations.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # 7. Block pattern +1+1-1-1
    initializations.append([1 if (i // 2) % 2 == 0 else -1 for i in range(n)])
    
    # 8. Block pattern +1+1+1-1-1-1
    initializations.append([1 if (i // 3) % 2 == 0 else -1 for i in range(n)])
    
    best_signs = None
    best_score = float('inf')
    
    for init in initializations:
        score = _max_abs_ap_sum(init)
        if score < best_score:
            best_score = score
            best_signs = init[:]
    
    current_signs = best_signs[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 60
    iteration = 0
    restart_interval = 1500
    
    # Pre-compute candidate list, update periodically
    candidates = _get_tight_ap_info(current_signs, top_k=40)
    
    while time.time() < deadline:
        iteration += 1
        
        if no_improve_count > max_no_improve or iteration % restart_interval == 0:
            if rng.random() < 0.4:
                current_signs = best_signs[:]
                for _ in range(rng.randint(3, 15)):
                    idx = rng.randint(0, n - 1)
                    current_signs[idx] = -current_signs[idx]
            else:
                current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            continue
        
        # Update candidates periodically
        if iteration % 30 == 0:
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            if not candidates:
                candidates = list(range(n))
        
        # Choose flip strategy: 80% single flip from tight indices, 20% random or 2-flip
        if rng.random() < 0.8:
            # Single flip from tight AP indices
            if candidates:
                idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                idx = rng.randint(0, n - 1)
            
            old_val = current_signs[idx]
            current_signs[idx] = -old_val
            new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                current_signs[idx] = old_val
                no_improve_count += 1
        else:
            # Try 2-flip move: flip two indices simultaneously
            if candidates and len(candidates) >= 2:
                i1 = candidates[rng.randint(0, len(candidates) - 1)]
                i2 = candidates[rng.randint(0, len(candidates) - 1)]
                if i1 == i2:
                    i2 = (i2 + 1) % n
            else:
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % n
            
            v1 = current_signs[i1]
            v2 = current_signs[i2]
            current_signs[i1] = -v1
            current_signs[i2] = -v2
            new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                current_signs[i1] = v1
                current_signs[i2] = v2
                no_improve_count += 1
    
    return best_signs
