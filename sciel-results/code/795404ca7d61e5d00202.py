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
            idx = np.arange(s, n, d)
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    
    return max_sum

def _get_tight_ap_indices(signs, top_k=60):
    """
    Find indices that appear in APs with sums close to the maximum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    ap_sums = []
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > 0:
                ap_sums.append((current_max, s, d))
            if current_max > max_sum:
                max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    tight_threshold = max(max_sum - 2, 0)
    tight_aps = [(s, d) for (val, s, d) in ap_sums if val >= tight_threshold]
    
    idx_count = np.zeros(n, dtype=np.int32)
    for s, d in tight_aps:
        idx = s
        while idx < n:
            idx_count[idx] += 1
            idx += d
    
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()

def _evaluate_flip_gain(signs, idx, current_max_sum):
    """
    Estimate how much the max AP sum changes if we flip signs[idx].
    Returns the new max AP sum (or an estimate).
    """
    n = len(signs)
    signs_arr = signs.copy()
    signs_arr[idx] = -signs_arr[idx]
    
    # Only check APs that contain idx
    new_max = 0
    for d in range(1, n + 1):
        # Find APs with step d that contain idx
        # AP is (s, s+d, s+2d, ...) where s ≡ idx (mod d) and s in [0, d-1]
        s0 = idx % d
        if s0 >= n:
            continue
        idx_arr = np.arange(s0, n, d)
        if idx not in idx_arr:
            continue
        cumsum = np.cumsum(np.asarray(signs_arr)[idx_arr])
        current_max = int(np.max(cumsum) - np.min(cumsum))
        if current_max > new_max:
            new_max = current_max
        # Early exit if already above current max
        if new_max > current_max_sum:
            break
    
    # Also need to check APs that don't contain idx but might have been max
    # But if we're trying to reduce max, and new_max <= current_max_sum,
    # then the non-containing APs are unchanged, so overall max is max(new_max, old_max_of_non_containing)
    # We don't track old_max_of_non_containing separately, so just return new_max as lower bound
    # Actually this is tricky. Let's just do a full check if new_max <= current_max_sum
    if new_max <= current_max_sum:
        return _max_abs_ap_sum(signs_arr, threshold=current_max_sum)
    else:
        return new_max

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    rng = random.Random(123)
    
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
    
    # 9. Greedy construction
    greedy_signs = [0] * n
    for i in range(n):
        greedy_signs[i] = 1
        max_plus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy_signs[s]
                max_plus = max(max_plus, abs(total))
                s -= d
        greedy_signs[i] = -1
        max_minus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy_signs[s]
                max_minus = max(max_minus, abs(total))
                s -= d
        if max_plus <= max_minus:
            greedy_signs[i] = 1
        else:
            greedy_signs[i] = -1
    initializations.append(greedy_signs)
    
    # 10. Greedy variant
    greedy2_signs = [0] * n
    for i in range(n):
        greedy2_signs[i] = 1
        max_plus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy2_signs[s]
                max_plus = max(max_plus, abs(total))
                s -= d
        greedy2_signs[i] = -1
        max_minus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy2_signs[s]
                max_minus = max(max_minus, abs(total))
                s -= d
        if max_plus < max_minus:
            greedy2_signs[i] = 1
        else:
            greedy2_signs[i] = -1
    initializations.append(greedy2_signs)
    
    # 11. New: Walsh-like pattern using bit-reversal
    walsh = []
    for i in range(n):
        # Use a different bit-reversal pattern
        r = 0
        x = i
        for b in range(8):
            if x & (1 << b):
                r |= (1 << (7 - b))
        walsh.append(1 if bin(r).count('1') % 2 == 0 else -1)
    initializations.append(walsh)
    
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
    max_no_improve = 80
    iteration = 0
    restart_interval = 1200
    
    # Pre-compute candidate list
    candidates = _get_tight_ap_indices(current_signs, top_k=60)
    
    while time.time() < deadline:
        iteration += 1
        
        if no_improve_count > max_no_improve or iteration % restart_interval == 0:
            if rng.random() < 0.5:
                current_signs = best_signs[:]
                for _ in range(rng.randint(5, 20)):
                    idx = rng.randint(0, n - 1)
                    current_signs[idx] = -current_signs[idx]
            else:
                current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            candidates = _get_tight_ap_indices(current_signs, top_k=60)
            continue
        
        # Update candidates periodically
        if iteration % 25 == 0:
            candidates = _get_tight_ap_indices(current_signs, top_k=60)
            if not candidates:
                candidates = list(range(n))
        
        # Choose flip strategy
        r = rng.random()
        if r < 0.7:
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
        elif r < 0.85:
            # 2-flip move
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
        else:
            # 3-flip move: flip three indices
            if candidates and len(candidates) >= 3:
                i1 = candidates[rng.randint(0, len(candidates) - 1)]
                i2 = candidates[rng.randint(0, len(candidates) - 1)]
                i3 = candidates[rng.randint(0, len(candidates) - 1)]
                # Ensure distinct
                if i1 == i2:
                    i2 = (i2 + 1) % n
                if i1 == i3:
                    i3 = (i3 + 2) % n
                if i2 == i3:
                    i3 = (i3 + 1) % n
            else:
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                i3 = rng.randint(0, n - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % n
                if i1 == i3:
                    i3 = (i3 + 2) % n
                if i2 == i3:
                    i3 = (i3 + 1) % n
            
            v1 = current_signs[i1]
            v2 = current_signs[i2]
            v3 = current_signs[i3]
            current_signs[i1] = -v1
            current_signs[i2] = -v2
            current_signs[i3] = -v3
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
                current_signs[i3] = v3
                no_improve_count += 1
    
    return best_signs
