# EVOLVE-BLOCK-START
import time
import random

def _can_add(S, cand, bitset):
    """
    Check if cand can be added to S without violating the mean condition.
    S is a list of existing elements (not including cand).
    bitset is a list of bitsets for subset sums of S (already computed).
    Returns True if cand can be safely added.
    """
    n = len(S)
    if n < 1:
        return True
    
    max_sum = n * 512
    
    # Check 1: Is cand the mean of any subset of S with size >= 2?
    # Need: exists k >= 2 such that bit (k * cand) is set in bitset[k]
    for k in range(2, n + 1):
        target = k * cand
        if target <= max_sum and (bitset[k] >> target) & 1:
            return False
    
    # Check 2: For each a in S, is a the mean of a subset including cand?
    # U = {cand} ∪ T where T ⊆ S\{a}, |T| >= 1, cand + sum(T) = a*(|T|+1)
    # sum(T) = a*(k+1) - cand where k = |T| >= 1
    for a in S:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            if 0 <= target <= max_sum and (bitset[k] >> target) & 1:
                return False
    
    return True

def _update_bitset(bitset, cand):
    """
    Update bitset to include cand.
    bitset is the current list of bitsets for the existing set.
    Returns new bitset list.
    """
    n_old = len(bitset) - 1
    n = n_old + 1
    new_bitset = [0] * (n + 1)
    new_bitset[0] = 1
    
    for k in range(n_old + 1):
        new_bitset[k] = bitset[k]
    
    for k in range(n, 0, -1):
        if k - 1 <= n_old:
            new_bitset[k] |= (bitset[k - 1] << cand)
    
    return new_bitset

def _conflict_score(x, all_numbers):
    """
    Refined conflict score: count 2-element arithmetic progression conflicts.
    """
    count_2 = 0
    for a in all_numbers:
        if a == x:
            continue
        b = 2 * x - a
        if b != x and b != a and b in all_numbers:
            count_2 += 1
    
    # Add centrality penalty
    centrality = abs(x - 256)
    score = count_2 * 10 + (512 - centrality) // 10
    
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    # Precompute conflict scores for all numbers 1..512
    all_numbers = set(range(1, 513))
    conflict_scores = {}
    for x in range(1, 513):
        conflict_scores[x] = _conflict_score(x, all_numbers)
    
    # Sort candidates by conflict score (fewer conflicts first)
    candidates_sorted = sorted(range(1, 513), key=lambda x: conflict_scores[x])
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Choose candidate ordering
        if restart == 0:
            candidates = list(candidates_sorted)
        elif restart == 1:
            candidates = list(reversed(candidates_sorted))
        elif restart == 2:
            candidates = list(range(1, 513))
            random.shuffle(candidates)
        elif restart % 3 == 0:
            # Perturbed sorted order
            candidates = list(candidates_sorted)
            for _ in range(50):
                i = random.randint(0, len(candidates) - 2)
                candidates[i], candidates[i+1] = candidates[i+1], candidates[i]
        else:
            # Random with slight bias toward low conflict
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: conflict_scores[x] + random.random() * 50)
        
        current_set = []
        current_set_set = set()
        bitset = [1]  # dp[0] = 1, no elements, sum 0
        
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            
            if cand in current_set_set:
                continue
            
            if _can_add(current_set, cand, bitset):
                current_set.append(cand)
                current_set_set.add(cand)
                bitset = _update_bitset(bitset, cand)
        
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
            
            if best_len >= 48:
                return best_set
    
    # Try to add any remaining numbers to best_set
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        remaining.sort(key=lambda x: conflict_scores[x])
        
        # Recompute bitset for best_set properly
        bitset = [1]
        for x in best_set:
            bitset = _update_bitset(bitset, x)
        
        for cand in remaining:
            if time.time() > deadline - 2.0:
                break
            if _can_add(best_set, cand, bitset):
                best_set.append(cand)
                bitset = _update_bitset(bitset, cand)
    
    return best_set

# EVOLVE-BLOCK-END
