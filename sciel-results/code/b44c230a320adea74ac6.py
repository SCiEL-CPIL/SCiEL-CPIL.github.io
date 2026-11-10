import time
from fractions import Fraction

def _is_subset_sum(target, others):
    """Check if target is a subset sum of reciprocals of others."""
    achievable = {Fraction(0, 1)}
    for y in others:
        new_sums = set()
        for a in achievable:
            ns = a + Fraction(1, y)
            if ns <= 1:
                new_sums.add(ns)
        achievable.update(new_sums)
    return target in achievable

def _is_valid(s):
    """Check if set s is reciprocal-subsum-free."""
    s_list = list(s)
    for x in s_list:
        target = Fraction(1, x)
        others = [y for y in s_list if y != x]
        if _is_subset_sum(target, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Greedy construction from largest to smallest
    achievable_sums = {Fraction(0, 1)}
    current_set = []
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, x)
        
        if target in achievable_sums:
            continue
        
        current_set.append(x)
        
        new_sums = set()
        for s in achievable_sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        achievable_sums.update(new_sums)
        
    if not current_set:
        return [1]
    
    # Verify the full set and repair if needed
    if not _is_valid(current_set):
        # Iteratively remove elements that are subsummed
        changed = True
        max_iters = 50
        iters = 0
        while changed and iters < max_iters and time.time() < deadline:
            changed = False
            iters += 1
            for x in list(current_set):
                if time.time() > deadline:
                    break
                target = Fraction(1, x)
                others = [y for y in current_set if y != x]
                if _is_subset_sum(target, others):
                    current_set.remove(x)
                    changed = True
                    break  # restart scan after removal
    
    if not current_set:
        return [1]
    
    return current_set
