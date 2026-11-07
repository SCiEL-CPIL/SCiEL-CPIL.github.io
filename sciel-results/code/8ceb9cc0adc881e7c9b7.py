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
    added = [0] * (n + 1)
    for k in range(n + 1):
        if sums_by_k[k]:
            added[k] = sums_by_k[k] << c
    
    # Build the new sums_by_k array first
    new_sums = [0] * (n + 2)
    for k in range(n + 1):
        if sums_by_k[k]:
            new_sums[k] |= sums_by_k[k]
            new_sums[k+1] |= added[k]
    
    # Check for collisions between different cardinalities
    for i in range(n + 2):
        if not new_sums[i]:
            continue
        for j in range(i + 1, n + 2):
            if new_sums[j] and (new_sums[i] & new_sums[j]):
                return None
    
    return new_sums

def _score_sums(sums_by_k):
    """Score: lower is better. Penalize total spread and concentration."""
    total_bits = 0
    max_bit = 0
    for s in sums_by_k:
        if s:
            total_bits += s.bit_count()
            max_bit = max(max_bit, s.bit_length())
    # Penalize large sums more heavily
    return total_bits + max_bit * 0.1

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

def _local_search(vals, max_iterations=200):
    """Try to improve the set by swapping elements."""
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
    while improved and time.time() - START < DEADLINE - 5:
        improved = False
        for i in range(n):
            if time.time() - START > DEADLINE - 5:
                break
            for new_val in available:
                # Try swapping current_vals[i] with new_val
                trial = current_vals[:]
                trial[i] = new_val
                trial_set = set(trial)
                if len(trial_set) != n:
                    continue
                
                # Check validity
                if _validate(trial):
                    # Compute score
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

def construct():
    global START, DEADLINE
    
    best_result = [1, 2, 4, 8, 16, 32, 64, 128]
    best_len = len(best_result)
    
    # Strategy 1: Descending greedy (proven effective) - more iterations
    for seed in range(80):
        if time.time() - START > DEADLINE - 8:
            break
        res = _greedy_descending(seed)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Strategy 2: Ascending greedy
    for seed in range(40):
        if time.time() - START > DEADLINE - 5:
            break
        res = _greedy_ascending(seed)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Strategy 3: Local search on best result
    if time.time() - START < DEADLINE - 12:
        improved = _local_search(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Strategy 4: More diverse restarts with mixed strategies
    for seed in range(30):
        if time.time() - START > DEADLINE - 3:
            break
        # Mix of descending and random
        rng = random.Random(seed * 7 + 13)
        if seed % 2 == 0:
            res = _greedy_descending(seed * 31)
        else:
            res = _greedy_ascending(seed * 47)
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
