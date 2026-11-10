import time
import random
import numpy as np

def _build_ap_structure(signs, n):
    """
    Build prefix-sum structure for all APs.
    For each step d and residue r (0 <= r < d), store the prefix sum along the AP.
    """
    signs_arr = np.asarray(signs, dtype=np.int8)
    prefixes = {}
    ap_max = {}
    ap_min = {}
    ap_range = {}
    global_max = 0
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                continue
            idx = np.arange(r, n, d)
            if len(idx) == 0:
                continue
            ps = np.zeros(len(idx) + 1, dtype=np.int32)
            ps[1:] = np.cumsum(signs_arr[idx])
            prefixes[(d, r)] = ps
            mx = int(np.max(ps))
            mn = int(np.min(ps))
            ap_max[(d, r)] = mx
            ap_min[(d, r)] = mn
            rng_val = mx - mn
            ap_range[(d, r)] = rng_val
            if rng_val > global_max:
                global_max = rng_val
    
    return prefixes, ap_max, ap_min, ap_range, global_max

def _apply_flip(signs, p, prefixes, ap_max, ap_min, ap_range, n):
    """
    Apply flip at position p. Returns (new_global_max, old_sign).
    """
    old_sign = signs[p]
    delta = -2 * old_sign
    signs[p] = -old_sign
    
    for d in range(1, n + 1):
        r = p % d
        if r >= n:
            continue
        key = (d, r)
        if key not in prefixes:
            continue
        ps = prefixes[key]
        L = len(ps) - 1
        k = (p - r) // d
        if k >= L:
            continue
        ps[k+1:] += delta
        ap_max[key] = int(np.max(ps))
        ap_min[key] = int(np.min(ps))
        ap_range[key] = ap_max[key] - ap_min[key]
    
    new_global_max = max(ap_range.values()) if ap_range else 0
    return new_global_max, old_sign

def _undo_flip(signs, p, prefixes, ap_max, ap_min, ap_range, n, old_sign):
    """
    Undo flip at position p, restoring old_sign.
    """
    delta = -2 * old_sign
    signs[p] = old_sign
    for d in range(1, n + 1):
        r = p % d
        if r >= n:
            continue
        key = (d, r)
        if key not in prefixes:
            continue
        ps = prefixes[key]
        L = len(ps) - 1
        k = (p - r) // d
        if k >= L:
            continue
        ps[k+1:] -= delta
        ap_max[key] = int(np.max(ps))
        ap_min[key] = int(np.min(ps))
        ap_range[key] = ap_max[key] - ap_min[key]

def _max_abs_ap_sum(signs, threshold=None):
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_sum = 0
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            if len(idx) == 0:
                continue
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    return max_sum

def _get_tight_ap_info(signs, top_k=40):
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    ap_sums = []
    max_sum = 0
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            if len(idx) == 0:
                continue
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
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
    
    initializations = []
    initializations.append([1 if i % 2 == 0 else -1 for i in range(n)])
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    initializations.append(tm)
    for _ in range(4):
        initializations.append([rng.choice((-1, 1)) for _ in range(n)])
    initializations.append([1 if (i // 2) % 2 == 0 else -1 for i in range(n)])
    initializations.append([1 if (i // 3) % 2 == 0 else -1 for i in range(n)])
    
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
        greedy_signs[i] = 1 if max_plus <= max_minus else -1
    initializations.append(greedy_signs)
    
    best_signs = None
    best_score = float('inf')
    for init in initializations:
        score = _max_abs_ap_sum(init)
        if score < best_score:
            best_score = score
            best_signs = init[:]
    
    current_signs = best_signs[:]
    current_score = best_score
    prefixes, ap_max, ap_min, ap_range, _ = _build_ap_structure(current_signs, n)
    
    no_improve_count = 0
    max_no_improve = 60
    iteration = 0
    restart_interval = 1500
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
            prefixes, ap_max, ap_min, ap_range, _ = _build_ap_structure(current_signs, n)
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            continue
        
        if iteration % 30 == 0:
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            if not candidates:
                candidates = list(range(n))
        
        if rng.random() < 0.8:
            if candidates:
                idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                idx = rng.randint(0, n - 1)
            
            new_score, old_sign = _apply_flip(current_signs, idx, prefixes, ap_max, ap_min, ap_range, n)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                _undo_flip(current_signs, idx, prefixes, ap_max, ap_min, ap_range, n, old_sign)
                no_improve_count += 1
        else:
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
            
            _, old1 = _apply_flip(current_signs, i1, prefixes, ap_max, ap_min, ap_range, n)
            _, old2 = _apply_flip(current_signs, i2, prefixes, ap_max, ap_min, ap_range, n)
            new_score = max(ap_range.values()) if ap_range else 0
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                _undo_flip(current_signs, i2, prefixes, ap_max, ap_min, ap_range, n, old2)
                _undo_flip(current_signs, i1, prefixes, ap_max, ap_min, ap_range, n, old1)
                no_improve_count += 1
    
    return best_signs
