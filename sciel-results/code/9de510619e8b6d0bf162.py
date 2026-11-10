import time
from fractions import Fraction

def _is_subset_sum(target, others):
    """Check if target is a subset sum of reciprocals of others."""
    if not others:
        return target == Fraction(0, 1)
    achievable = {Fraction(0, 1)}
    for y in others:
        new_sums = set()
        for a in achievable:
            ns = a + Fraction(1, y)
            if ns <= target:
                new_sums.add(ns)
        if target in new_sums:
            return True
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

def _can_add(s, x):
    """Check if adding x to set s maintains validity."""
    s_list = list(s)
    # Check: is 1/x a subset sum of reciprocals of s?
    if _is_subset_sum(Fraction(1, x), s_list):
        return False
    # Check: for each y in s, is 1/y a subset sum of reciprocals of s + {x} minus {y}?
    for y in s_list:
        target = Fraction(1, y)
        others = [z for z in s_list if z != y] + [x]
        if _is_subset_sum(target, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Try multiple greedy orderings and local search
    best_set = []
    
    orderings = [
        list(range(24, 0, -1)),  # largest to smallest
        list(range(1, 25)),      # smallest to largest
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        
        achievable_sums = {Fraction(0, 1)}
        current_set = []
        
        for x in ordering:
            if time.time() > deadline - 5:
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
        
        # Verify and repair
        if not _is_valid(current_set):
            changed = True
            max_iters = 50
            iters = 0
            while changed and iters < max_iters and time.time() < deadline - 3:
                changed = False
                iters += 1
                for x in list(current_set):
                    if time.time() > deadline - 3:
                        break
                    target = Fraction(1, x)
                    others = [y for y in current_set if y != x]
                    if _is_subset_sum(target, others):
                        current_set.remove(x)
                        changed = True
                        break
        
        # Local search: try to add elements back
        if time.time() < deadline - 3:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for x in range(1, 25):
                    if time.time() > deadline - 2:
                        break
                    if x not in current_set:
                        if _can_add(current_set, x):
                            current_set.append(x)
                            improved = True
                            break
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
    
    if not best_set:
        return [1]
    
    return best_set
