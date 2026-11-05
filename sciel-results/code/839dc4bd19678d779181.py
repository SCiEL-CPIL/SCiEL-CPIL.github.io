"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys

def _is_valid_triples(S):
    """Check if all unordered triple sums (with replacement) are unique."""
    n = len(S)
    seen = set()
    # (i, j, k) with i <= j <= k
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _check_conflict(x, S, existing_sums):
    """
    Check if adding x to S creates any conflict.
    Returns (valid, new_sums_set) where new_sums_set is the set of new triple sums.
    """
    n = len(S)
    new_sums = set()
    
    # Check (x,x,x)
    s = 3 * x
    if s in existing_sums:
        return False, None
    if s in new_sums:
        return False, None
    new_sums.add(s)
    
    # Check (x,x,y) for y in S
    for y in S:
        s = 2 * x + y
        if s in existing_sums:
            return False, None
        if s in new_sums:
            return False, None
        new_sums.add(s)
    
    # Check (x,y,y) for y in S
    for y in S:
        s = x + 2 * y
        if s in existing_sums:
            return False, None
        if s in new_sums:
            return False, None
        new_sums.add(s)
    
    # Check (x,y,z) for y < z in S
    for i in range(n):
        for j in range(i + 1, n):
            s = x + S[i] + S[j]
            if s in existing_sums:
                return False, None
            if s in new_sums:
                return False, None
            new_sums.add(s)
    
    return True, new_sums

def _greedy_build(candidate_order, deadline=None):
    """
    Build a valid set using greedy insertion with given candidate order.
    """
    S = []
    existing_sums = set()
    
    for x in candidate_order:
        if deadline and time.time() > deadline:
            break
            
        valid, new_sums = _check_conflict(x, S, existing_sums)
        
        if valid:
            S.append(x)
            existing_sums.update(new_sums)
            
    return S

def _generate_weighted_candidates(seed, n_max=4096, bias=0.7):
    """
    Generate candidates with a bias toward smaller numbers.
    Smaller numbers tend to create fewer conflicts.
    """
    rng = random.Random(seed)
    candidates = list(range(1, n_max + 1))
    
    # Create weights that favor smaller numbers
    weights = []
    for i, c in enumerate(candidates):
        # Exponential decay: smaller numbers get higher weight
        w = bias ** i
        weights.append(w)
    
    # Sample without replacement using weighted selection
    selected = []
    remaining = list(range(len(candidates)))
    remaining_weights = weights[:]
    
    for _ in range(len(candidates)):
        if not remaining:
            break
        total_w = sum(remaining_weights)
        if total_w <= 0:
            # Fallback to uniform
            idx = rng.randint(0, len(remaining) - 1)
        else:
            r = rng.random() * total_w
            cumsum = 0
            idx = 0
            for i, w in enumerate(remaining_weights):
                cumsum += w
                if r <= cumsum:
                    idx = i
                    break
            else:
                idx = len(remaining) - 1
        
        selected.append(candidates[remaining[idx]])
        remaining.pop(idx)
        remaining_weights.pop(idx)
    
    return selected

def _local_extend(S, deadline):
    """
    Try to extend S by carefully adding elements.
    Uses a priority queue approach: try candidates in order of 'least conflict potential'.
    """
    S = list(S)
    existing_sums = set()
    
    # Rebuild existing_sums
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                existing_sums.add(S[i] + S[j] + S[k])
    
    # Try to add elements from 1..4096 that aren't already in S
    in_S = set(S)
    candidates = [x for x in range(1, 4097) if x not in in_S]
    
    # Sort candidates by value (smaller first, as they're less likely to conflict)
    candidates.sort()
    
    added = 0
    for x in candidates:
        if deadline and time.time() > deadline:
            break
            
        valid, new_sums = _check_conflict(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums.update(new_sums)
            added += 1
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_S = []
    
    # 1. Deterministic greedy (ascending order - baseline)
    S_det = _greedy_build(list(range(1, 4097)), deadline)
    if len(S_det) > len(best_S):
        best_S = S_det
    
    # 2. Try descending order
    S_desc = _greedy_build(list(range(4096, 0, -1)), deadline)
    if len(S_desc) > len(best_S):
        best_S = S_desc
    
    # 3. Weighted random restarts with bias toward smaller numbers
    attempts = 0
    best_size = len(best_S)
    no_improve_count = 0
    
    while time.time() < deadline:
        seed = random.randint(0, 10**9)
        
        # Use weighted candidates biased toward smaller numbers
        candidates = _generate_weighted_candidates(seed, bias=0.75)
        S_rand = _greedy_build(candidates, deadline)
        
        # Try to extend the result
        S_extended = _local_extend(S_rand, deadline)
        
        if len(S_extended) > len(best_S):
            best_S = S_extended
            best_size = len(best_S)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        attempts += 1
        
        # If we've gone many attempts without improvement, try different bias
        if no_improve_count > 50:
            # Change bias strategy
            bias = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            candidates = _generate_weighted_candidates(seed, bias=bias)
            S_rand = _greedy_build(candidates, deadline)
            S_extended = _local_extend(S_rand, deadline)
            
            if len(S_extended) > len(best_S):
                best_S = S_extended
                best_size = len(best_S)
            no_improve_count = 0
    
    return best_S

if __name__ == "__main__":
    # Quick test
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s}, Size: {len(res)}")
    # Verify
    if _is_valid_triples(res):
        print("Valid")
    else:
        print("Invalid")
