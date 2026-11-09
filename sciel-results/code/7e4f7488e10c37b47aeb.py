"""Search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import sys
from math import gcd
from functools import reduce

# LCM of 1..22
def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, range(1, 23))
# LCM = 232792560

# Weights: w_i = LCM // i for i in 1..22
_WEIGHTS = [None] + [_LCM // i for i in range(1, 23)]

def _verify(subset):
    """Check if all 2^k subset sums of weights are distinct."""
    k = len(subset)
    sums = {0}
    for i in subset:
        w = _WEIGHTS[i]
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _check_pair_conflict(i, j, existing_sums):
    """Check if adding both i and j to existing_sums causes a collision."""
    wi = _WEIGHTS[i]
    wj = _WEIGHTS[j]
    
    # Check if wi conflicts with existing_sums
    for s in existing_sums:
        if (s + wi) in existing_sums:
            return True
    # Check if wj conflicts with existing_sums
    for s in existing_sums:
        if (s + wj) in existing_sums:
            return True
    # Check if wi and wj conflict with each other
    for s in existing_sums:
        if (s + wi) in {x + wj for x in existing_sums}:
            return True
        if (s + wj) in {x + wi for x in existing_sums}:
            return True
    return False

def _can_add(weight, sums):
    """Check if adding weight to sums causes any collision."""
    for s in sums:
        if (s + weight) in sums:
            return False
    return True

def _new_sums_after_add(weight, sums):
    """Compute new sums after adding weight."""
    return sums | {s + weight for s in sums}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Start with a known good solution
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Try to find larger solutions using MIS-inspired branch and bound
    # Strategy: build up from small sets, trying to add elements that don't conflict
    
    all_elements = list(range(1, 23))
    
    # Sort by weight (ascending) to try smaller weights first
    all_elements.sort(key=lambda x: _WEIGHTS[x])
    
    # Try different starting points and growth strategies
    for start_size in range(1, 10):
        if time.time() > deadline:
            break
        
        # Try all combinations of start_size elements
        from itertools import combinations
        
        for combo in combinations(all_elements, start_size):
            if time.time() > deadline:
                break
            
            # Check if this base set is valid
            base_sums = {0}
            valid = True
            for idx in combo:
                w = _WEIGHTS[idx]
                if not _can_add(w, base_sums):
                    valid = False
                    break
                base_sums = _new_sums_after_add(w, base_sums)
            
            if not valid:
                continue
            
            # Try to grow this set greedily
            current = list(combo)
            current_sums = base_sums
            
            # Try adding remaining elements
            for idx in all_elements:
                if idx in current:
                    continue
                w = _WEIGHTS[idx]
                if _can_add(w, current_sums):
                    current.append(idx)
                    current_sums = _new_sums_after_add(w, current_sums)
            
            if len(current) > len(best):
                best = current
        
        # If we found a solution of size >= 10, we can stop early
        if len(best) >= 10:
            break
    
    # Try local search: swap elements to potentially get larger sets
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_set = set(best)
        best_sums = {0}
        for idx in best:
            best_sums = _new_sums_after_add(_WEIGHTS[idx], best_sums)
        
        # Try swapping one element for another
        for remove_idx in best:
            # Remove remove_idx
            remaining = [x for x in best if x != remove_idx]
            rem_sums = {0}
            for idx in remaining:
                rem_sums = _new_sums_after_add(_WEIGHTS[idx], rem_sums)
            
            # Try adding each other element
            for add_idx in all_elements:
                if add_idx in best_set:
                    continue
                w = _WEIGHTS[add_idx]
                if _can_add(w, rem_sums):
                    new_set = remaining + [add_idx]
                    new_sums = _new_sums_after_add(w, rem_sums)
                    
                    # Try to grow further
                    current = new_set
                    current_sums = new_sums
                    current_set = set(current)
                    
                    for idx in all_elements:
                        if idx in current_set:
                            continue
                        w2 = _WEIGHTS[idx]
                        if _can_add(w2, current_sums):
                            current.append(idx)
                            current_sums = _new_sums_after_add(w2, current_sums)
                    
                    if len(current) > len(best):
                        best = current
                        improved = True
                        break
            if improved:
                break
    
    return best
