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
    prefix = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(signs_arr, out=prefix[1:])
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            # Indices in AP: s, s+d, s+2d, ... < n
            # Use prefix sums to compute cumulative sums along the AP
            # The AP has indices s, s+d, s+2d, ...
            # We need cumsum of signs_arr[s], signs_arr[s+d], ...
            # Using prefix: sum from s to s+kd = prefix[s+(k+1)*d] - prefix[s]
            # But we need the range of partial sums, which requires
            # computing all prefix differences
            # For efficiency, compute the AP values via slicing
            idx = np.arange(s, n, d)
            if len(idx) <= 1:
                continue
            # Compute cumulative sum along AP using prefix
            # cumsum_k = prefix[s + (k+1)*d] - prefix[s]
            # So cumsum values are: prefix[s+d]-prefix[s], prefix[s+2d]-prefix[s], ...
            # The max absolute sum on this AP = max(cumsum) - min(cumsum)
            # where cumsum includes 0 (empty prefix)
            # Actually, the max discrepancy on an AP is max over all sub-APs
            # which equals max(cumsum) - min(cumsum) where cumsum starts at 0
            
            # Get prefix values at positions s, s+d, s+2d, ...
            # prefix[s] is the sum before index s
            # prefix[s+d] - prefix[s] = sum of signs[s:s+d]
            # prefix[s+2d] - prefix[s] = sum of signs[s:s+2d]
            # etc.
            # So the cumulative sums along the AP (starting from 0) are:
            # 0, prefix[s+d]-prefix[s], prefix[s+2d]-prefix[s], ...
            
            # Number of terms in AP
            k = (n - 1 - s) // d + 1
            if k <= 1:
                continue
            # prefix positions: s, s+d, s+2d, ..., s+(k-1)*d
            # We want prefix[s + j*d] for j = 0, 1, ..., k-1
            # The cumulative sum at position j is prefix[s + j*d] - prefix[s]
            # So max cumsum - min cumsum = max(prefix[s+j*d]) - min(prefix[s+j*d])
            # (since prefix[s] is constant, it cancels in the difference)
            
            # Extract prefix values at these positions
            pos = np.arange(s, n, d)
            vals = prefix[pos]
            current_max = int(np.max(vals) - np.min(vals))
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
    prefix = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(signs_arr, out=prefix[1:])
    
    ap_sums = []
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            vals = prefix[s:n:d]
            current_max = int(vals.max() - vals.min())
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
    
    # 9. Greedy construction: at each position, choose sign that minimizes
    #    the max absolute AP sum for APs ending at that position
    greedy_signs = [0] * n
    for i in range(n):
        # Try +1
        greedy_signs[i] = 1
        # Compute max abs sum for APs ending at position i
        max_plus = 0
        for d in range(1, i + 1):
            # AP: i-d, i-2d, ..., i
            s = i
            total = 0
            while s >= 0:
                total += greedy_signs[s]
                max_plus = max(max_plus, abs(total))
                s -= d
        # Try -1
        greedy_signs[i] = -1
        max_minus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy_signs[s]
                max_minus = max(max_minus, abs(total))
                s -= d
        # Choose the better one
        if max_plus <= max_minus:
            greedy_signs[i] = 1
        else:
            greedy_signs[i] = -1
    initializations.append(greedy_signs)
    
    # 10. Another greedy variant with different tie-breaking
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
    max_no_improve = 40
    iteration = 0
    restart_interval = 800
    
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
        if iteration % 15 == 0:
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            if not candidates:
                candidates = list(range(n))
        
        # Try multiple single flips in sequence (first-improvement over candidates)
        found_improvement = False
        num_candidates = min(len(candidates), 12)  # Try up to 12 candidates per iteration
        if num_candidates > 0:
            # Shuffle a subset of candidates to try
            trial_indices = rng.sample(candidates, num_candidates)
            for idx in trial_indices:
                if time.time() > deadline:
                    break
                old_val = current_signs[idx]
                current_signs[idx] = -old_val
                new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                    found_improvement = True
                    if new_score < best_score:
                        best_score = new_score
                        best_signs = current_signs[:]
                    # Continue trying more flips after improvement
                else:
                    current_signs[idx] = old_val
        
        if not found_improvement:
            # No improvement from candidate flips; try random flip or 2-flip
            if rng.random() < 0.5:
                idx = rng.randint(0, n - 1)
                old_val = current_signs[idx]
                current_signs[idx] = -old_val
                new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                    found_improvement = True
                    if new_score < best_score:
                        best_score = new_score
                        best_signs = current_signs[:]
                else:
                    current_signs[idx] = old_val
            else:
                # 2-flip
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
                    found_improvement = True
                    if new_score < best_score:
                        best_score = new_score
                        best_signs = current_signs[:]
                else:
                    current_signs[i1] = v1
                    current_signs[i2] = v2
            
            if not found_improvement:
                no_improve_count += 1
    
    return best_signs
