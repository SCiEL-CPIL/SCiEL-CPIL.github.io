import time
import random

START = time.time()
DEADLINE = 55.0

def _validate(vals):
    n = len(vals)
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _add_element(sums_by_k, c):
    n = len(sums_by_k) - 1
    # Compute new sums for each cardinality
    new_sums = sums_by_k[:] + [0]
    for k in range(n, -1, -1):
        if sums_by_k[k]:
            new_sums[k+1] |= (sums_by_k[k] << c)
    
    # Check for conflicts: for each pair (i,j) with i != j, verify no overlap
    # Key insight: only need to check if new_sums[i] & new_sums[j] for i != j
    # But we can be smarter: only check pairs where at least one is new
    for i in range(n + 2):
        for j in range(i + 1, n + 2):
            if i == j:
                continue
            if new_sums[i] & new_sums[j]:
                return None
    
    return new_sums

def _score_sums(sums_by_k):
    """Score: lower is better. Penalize total bits and maximize slack."""
    total_bits = 0
    max_sum = 0
    for s in sums_by_k:
        if s:
            total_bits += s.bit_count()
            # Find highest set bit
            max_sum = max(max_sum, s.bit_length())
    # Lower total_bits is better (more compact)
    # But we also want to ensure we haven't used too much range
    # Use a weighted combination
    return total_bits + max_sum * 0.1

def _greedy_descending(seed, max_candidates=256):
    """Greedy best-first with descending bias."""
    rng = random.Random(seed)
    # Start with descending order, then shuffle slightly for diversity
    candidates = list(range(256, 0, -1))
    # Apply a light shuffle to break ties but maintain descending bias
    for i in range(len(candidates) - 1, 0, -1):
        # With decreasing probability, swap with a random earlier position
        if rng.random() < 0.1:  # 10% chance of disruption
            j = rng.randint(0, i)
            candidates[i], candidates[j] = candidates[j], candidates[i]
    
    sums_by_k = [1]
    current_vals = []
    
    while True:
        if time.time() - START > DEADLINE - 2:
            break
            
        best_c = None
        best_sums = None
        best_score = float('inf')
        
        # Limit candidates examined per step for speed
        search_limit = min(len(candidates), 80)
        
        for idx in range(search_limit):
            c = candidates[idx]
            new_sums = _add_element(sums_by_k, c)
            if new_sums is not None:
                score = _score_sums(new_sums)
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_sums = new_sums
                    best_idx = idx
                    
        if best_c is None:
            break
            
        sums_by_k = best_sums
        current_vals.append(best_c)
        # Remove the chosen candidate
        candidates.pop(best_idx)
        
    return current_vals

def _greedy_ascending(seed):
    """Greedy best-first with ascending order (smaller numbers first)."""
    rng = random.Random(seed)
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    
    sums_by_k = [1]
    current_vals = []
    
    while True:
        if time.time() - START > DEADLINE - 2:
            break
            
        best_c = None
        best_sums = None
        best_score = float('inf')
        
        search_limit = min(len(candidates), 60)
        
        for idx in range(search_limit):
            c = candidates[idx]
            new_sums = _add_element(sums_by_k, c)
            if new_sums is not None:
                score = _score_sums(new_sums)
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_sums = new_sums
                    best_idx = idx
                    
        if best_c is None:
            break
            
        sums_by_k = best_sums
        current_vals.append(best_c)
        candidates.pop(best_idx)
        
    return current_vals

def _local_search(vals, max_iterations=500):
    """Try to improve the set by swapping elements. Incremental sums update."""
    if not vals:
        return vals
    
    current_vals = vals[:]
    n = len(current_vals)
    
    # Build sums_by_k for current set
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in current_vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    
    used = set(current_vals)
    available = [x for x in range(1, 257) if x not in used]
    current_score = _score_sums(sums_by_k)
    
    improved = True
    while improved and time.time() - START < DEADLINE - 3:
        improved = False
        for i in range(n):
            if time.time() - START > DEADLINE - 3:
                break
            old_val = current_vals[i]
            for new_val in available:
                if new_val == old_val:
                    continue
                # Try swapping current_vals[i] with new_val
                trial = current_vals[:]
                trial[i] = new_val
                trial_set = set(trial)
                if len(trial_set) != n:
                    continue
                
                # Check validity using _validate
                if _validate(trial):
                    # Compute new sums
                    ts = [0] * (n + 1)
                    ts[0] = 1
                    for v in trial:
                        for k in range(n - 1, -1, -1):
                            if ts[k]:
                                ts[k + 1] |= (ts[k] << v)
                    trial_score = _score_sums(ts)
                    
                    if trial_score < current_score:
                        current_vals = trial
                        used = set(trial)
                        available = [x for x in range(1, 257) if x not in used]
                        sums_by_k = ts
                        current_score = trial_score
                        improved = True
                        break
            if improved:
                break
    
    return current_vals

def _greedy_mixed(seed, bias=0.5):
    """Greedy with mixed ordering: bias towards descending but with randomization."""
    rng = random.Random(seed)
    candidates = list(range(256, 0, -1))
    # Apply partial shuffle based on bias
    for i in range(len(candidates) - 1, 0, -1):
        if rng.random() < bias:
            j = rng.randint(0, i)
            candidates[i], candidates[j] = candidates[j], candidates[i]
    
    sums_by_k = [1]
    current_vals = []
    
    while True:
        if time.time() - START > DEADLINE - 2:
            break
            
        best_c = None
        best_sums = None
        best_score = float('inf')
        
        search_limit = min(len(candidates), 100)
        
        for idx in range(search_limit):
            c = candidates[idx]
            new_sums = _add_element(sums_by_k, c)
            if new_sums is not None:
                score = _score_sums(new_sums)
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_sums = new_sums
                    best_idx = idx
                    
        if best_c is None:
            break
            
        sums_by_k = best_sums
        current_vals.append(best_c)
        candidates.pop(best_idx)
        
    return current_vals

def construct():
    global START, DEADLINE
    
    best_result = [1, 2, 4, 8, 16, 32, 64, 128]
    best_len = len(best_result)
    
    # Strategy 1: Descending greedy (proven effective)
    for seed in range(80):
        if time.time() - START > DEADLINE - 5:
            break
        res = _greedy_descending(seed)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Strategy 2: Ascending greedy
    for seed in range(50):
        if time.time() - START > DEADLINE - 5:
            break
        res = _greedy_ascending(seed)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Strategy 3: Mixed greedy with various biases
    for bias in [0.2, 0.3, 0.5, 0.7, 0.8]:
        for seed in range(30):
            if time.time() - START > DEADLINE - 5:
                break
            res = _greedy_mixed(seed * 100 + int(bias * 10), bias)
            if len(res) > best_len:
                if _validate(res):
                    best_result = res
                    best_len = len(res)
    
    # Strategy 4: Local search on best result
    if time.time() - START < DEADLINE - 10:
        improved = _local_search(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Strategy 5: More restarts after local search
    for seed in range(40):
        if time.time() - START > DEADLINE - 3:
            break
        if seed % 2 == 0:
            res = _greedy_descending(seed * 31 + 1000)
        elif seed % 3 == 0:
            res = _greedy_ascending(seed * 47 + 2000)
        else:
            res = _greedy_mixed(seed * 73 + 3000, 0.4)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Final local search
    if time.time() - START < DEADLINE - 5:
        improved = _local_search(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    return [int(x) for x in best_result]

if __name__ == "__main__":
    res = construct()
    print(f"Found size: {len(res)}")
    print(res)
    print("Valid:", _validate(res))
