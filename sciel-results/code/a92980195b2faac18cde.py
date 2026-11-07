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
            
    for k in range(n + 1):
        if not added[k]:
            continue
        target_class = k + 1
        for m in range(n + 2):
            if m == target_class:
                continue
            nm = sums_by_k[m] if m <= n else 0
            if m > 0 and m - 1 <= n:
                nm |= added[m - 1]
            if added[k] & nm:
                return None
                
    new_sums = sums_by_k[:]
    new_sums.append(0)
    for k in range(n, -1, -1):
        if sums_by_k[k]:
            new_sums[k+1] |= (sums_by_k[k] << c)
            
    return new_sums

def _score_sums(sums_by_k):
    """Score: lower is better. Penalize total spread and concentration."""
    total_bits = 0
    for s in sums_by_k:
        total_bits += s.bit_count()
    return total_bits

def _greedy_best_first(seed):
    rng = random.Random(seed)
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    
    sums_by_k = [1]
    current_vals = []
    
    while True:
        if time.time() - START > DEADLINE - 1:
            break
            
        best_c = None
        best_sums = None
        best_score = float('inf')
        
        for c in candidates:
            new_sums = _add_element(sums_by_k, c)
            if new_sums is not None:
                score = _score_sums(new_sums)
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_sums = new_sums
                    
        if best_c is None:
            break
            
        sums_by_k = best_sums
        current_vals.append(best_c)
        candidates.remove(best_c)
        
    return current_vals

def construct():
    global START, DEADLINE
    
    best_result = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Run multiple greedy best-first searches with different seeds
    for seed in range(200):
        if time.time() - START > DEADLINE - 1:
            break
        res = _greedy_best_first(seed)
        if len(res) > len(best_result):
            if _validate(res):
                best_result = res
                
    return [int(x) for x in best_result]

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found size: {len(res)}")
    print(res)
    print("Valid:", _validate(res))
