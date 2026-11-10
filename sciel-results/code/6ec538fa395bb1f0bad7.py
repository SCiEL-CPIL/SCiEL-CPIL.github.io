import time
import random
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _can_form_target(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals.
    vals is a list of integers.
    Uses meet-in-the-middle for efficiency.
    """
    if target == 0:
        return True
    if not vals:
        return False
    if target > sum(vals):
        return False
    
    n = len(vals)
    half = n // 2
    left = vals[:half]
    right = vals[half:]
    
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
        
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
        
    for s in left_sums:
        if target - s in right_sums:
            return True
    return False

def _is_valid_set(S):
    """
    Check if no element's reciprocal is the sum of reciprocals of a subset of the others.
    S is a list of distinct integers.
    """
    n = len(S)
    if n <= 1:
        return True
        
    for i, x in enumerate(S):
        others = S[:i] + S[i+1:]
        if not others:
            continue
            
        try:
            L = reduce(_lcm, others, 1)
        except ZeroDivisionError:
            continue
            
        if L % x != 0:
            continue
            
        target = L // x
        if target == 0:
            continue
            
        vals = [L // y for y in others]
        
        if _can_form_target(target, vals):
            return False
            
    return True

def _is_valid_pair(x, y, context):
    """
    Check if adding x to a set containing y and context is valid with respect to x and y.
    This is a partial check for pruning in branch and bound.
    context is a set of other elements.
    """
    # We need to check:
    # 1. Is 1/x a subset sum of reciprocals of {y} U context?
    # 2. Is 1/y a subset sum of reciprocals of {x} U context?
    
    # Check 1: 1/x = sum(1/z for z in subset of {y} U context)
    others_for_x = list({y} | context)
    if others_for_x:
        L = reduce(_lcm, others_for_x, 1)
        if L % x == 0:
            target = L // x
            if target > 0:
                vals = [L // z for z in others_for_x]
                if _can_form_target(target, vals):
                    return False
    
    # Check 2: 1/y = sum(1/z for z in subset of {x} U context)
    others_for_y = list({x} | context)
    if others_for_y:
        L = reduce(_lcm, others_for_y, 1)
        if L % y == 0:
            target = L // y
            if target > 0:
                vals = [L // z for z in others_for_y]
                if _can_form_target(target, vals):
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin
    
    best = list(range(13, 25))
    best_size = len(best)
    
    # Verify initial best
    if _is_valid_set(best):
        pass
    else:
        # Fallback if initial guess is invalid (shouldn't happen)
        best = [24]
        best_size = 1

    # Branch and Bound to find larger sets
    # We search for a subset of [1..24] that is valid and large.
    # To speed up, we can fix the "large" part and search for "small" additions.
    # Or we can do a full search with pruning.
    
    # Strategy: 
    # 1. Try to extend the current best set by adding elements from 1..12.
    # 2. Use B&B to find if we can add a combination of small elements.
    
    # Let's try a recursive search to find the largest valid set.
    # To make it feasible, we'll restrict the search space or use strong pruning.
    # Given N=24, 2^24 is too large. But we can prune heavily.
    
    # A better approach for B&B:
    # Start with an empty set. Add elements in descending order (24, 23, ..., 1).
    # Prune if current set size + remaining possible elements <= best_size.
    # Prune if current set is invalid.
    
    # However, checking validity at every node is expensive.
    # Let's stick to the "Extension" strategy which has been effective.
    
    # Improvement: Use local search on the "missing" elements.
    # Identify elements in 1..12 not in best.
    missing = [x for x in range(1, 13) if x not in best]
    
    # Try all subsets of missing elements of size 1, then 2, etc.
    # Since missing has at most 12 elements, we can try subsets up to size 3 or 4.
    # This is much faster than full B&B.
    
    # First, try adding single elements
    for x in missing:
        if time.time() > deadline:
            break
        test_set = sorted(list(set(best) | {x}))
        if _is_valid_set(test_set):
            if len(test_set) > best_size:
                best = test_set
                best_size = len(best)
                missing.remove(x)
    
    # Then, try adding pairs from remaining missing
    # Re-calculate missing
    missing = [x for x in range(1, 13) if x not in best]
    
    for i in range(len(missing)):
        for j in range(i+1, len(missing)):
            if time.time() > deadline:
                break
            x, y = missing[i], missing[j]
            test_set = sorted(list(set(best) | {x, y}))
            if _is_valid_set(test_set):
                if len(test_set) > best_size:
                    best = test_set
                    best_size = len(best)
                    missing.remove(x)
                    if y in missing:
                        missing.remove(y)
                    # Re-calculate to be safe
                    missing = [x for x in range(1, 13) if x not in best]
                    break
        if time.time() > deadline:
            break
    
    # Try adding triples
    missing = [x for x in range(1, 13) if x not in best]
    for i in range(len(missing)):
        for j in range(i+1, len(missing)):
            for k in range(j+1, len(missing)):
                if time.time() > deadline:
                    break
                x, y, z = missing[i], missing[j], missing[k]
                test_set = sorted(list(set(best) | {x, y, z}))
                if _is_valid_set(test_set):
                    if len(test_set) > best_size:
                        best = test_set
                        best_size = len(best)
                        for m in [x, y, z]:
                            if m in missing:
                                missing.remove(m)
                        missing = [x for x in range(1, 13) if x not in best]
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we still have time, try random restarts with greedy + local search
    if time.time() < deadline - 5.0:
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            
            # Random order of 1..24
            elems = list(range(1, 25))
            random.shuffle(elems)
            
            S = []
            for x in elems:
                test = S + [x]
                if _is_valid_set(test):
                    S = test
            
            if len(S) > best_size:
                best = S
                best_size = len(best)
                
            # Local search: try to add missing elements to S
            if len(S) < 24:
                missing_S = [x for x in range(1, 25) if x not in S]
                random.shuffle(missing_S)
                current_S = list(S)
                for x in missing_S:
                    test = current_S + [x]
                    if _is_valid_set(test):
                        current_S = test
                if len(current_S) > best_size:
                    best = current_S
                    best_size = len(best)

    return best
