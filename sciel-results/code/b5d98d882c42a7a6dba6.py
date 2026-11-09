"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

_LCM = 232792560

def _verify(subset):
    """Return True if all 2^len(subset) reciprocal subset sums are distinct."""
    k = len(subset)
    if k == 0:
        return True
    scaled = [_LCM // x for x in subset]
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    return len(set(sums)) == n

def _find_collision(subset):
    """Find two different subsets with the same sum, or None if dissociated."""
    k = len(subset)
    if k == 0:
        return None
    scaled = [_LCM // x for x in subset]
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    seen = {}
    for mask in range(n):
        s = sums[mask]
        if s in seen:
            return mask, seen[s]
        seen[s] = mask
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    universe = list(range(1, 23))
    
    # Phase 1: Greedy removal from full set
    current = list(range(1, 23))
    
    while True:
        if time.time() > deadline - 40:
            break
        col = _find_collision(current)
        if col is None:
            break
        mask1, mask2 = col
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        to_remove = max(sym_diff, key=lambda i: current[i])
        current.remove(current[to_remove])
    
    best = list(current)
    
    # Phase 2: Try to add back removed elements
    removed = [x for x in range(1, 23) if x not in best]
    for x in removed:
        if time.time() > deadline - 40:
            break
        candidate = best + [x]
        candidate.sort()
        if _verify(candidate):
            best = candidate
    
    # Phase 3: Local search with swaps to improve
    in_set = set(best)
    not_in_set = [x for x in range(1, 23) if x not in in_set]
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i, a in enumerate(best):
            if time.time() > deadline - 30:
                break
            for b in not_in_set:
                candidate = best[:]
                candidate[i] = b
                candidate.sort()
                if _verify(candidate):
                    best = candidate
                    in_set = set(best)
                    not_in_set = [x for x in range(1, 23) if x not in in_set]
                    improved = True
                    break
    
    # Phase 4: Try to find larger sets via randomized search
    random.seed(42)
    current_best_size = len(best)
    
    for target_size in range(current_best_size + 1, 23):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            candidate = random.sample(universe, target_size)
            candidate.sort()
            
            if _verify(candidate):
                best = candidate
                found = True
                break
            
            # Try to fix via local search
            in_cand = set(candidate)
            not_in_cand = [x for x in universe if x not in in_cand]
            
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                col = _find_collision(candidate)
                if col is None:
                    if len(candidate) > len(best):
                        best = candidate
                        found = True
                    break
                
                mask1, mask2 = col
                k = len(candidate)
                sym_diff = []
                for i in range(k):
                    bit = 1 << i
                    if (mask1 & bit) != (mask2 & bit):
                        sym_diff.append(i)
                
                swapped = False
                for idx in sym_diff:
                    if time.time() > deadline - 10:
                        break
                    for b in not_in_cand:
                        new_cand = candidate[:]
                        new_cand[idx] = b
                        new_cand.sort()
                        if _verify(new_cand):
                            candidate = new_cand
                            in_cand = set(candidate)
                            not_in_cand = [x for x in universe if x not in in_cand]
                            swapped = True
                            break
                    if swapped:
                        break
                
                if not swapped:
                    to_remove = max(sym_diff, key=lambda i: candidate[i])
                    candidate.remove(candidate[to_remove])
                    in_cand = set(candidate)
                    not_in_cand = [x for x in universe if x not in in_cand]
        
        if found:
            current_best_size = len(best)
            continue
        else:
            break
    
    return best
