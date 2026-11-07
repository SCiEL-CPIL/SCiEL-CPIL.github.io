import random
import time
import math

def _validate(elements):
    n = len(elements)
    if n == 0:
        return True
    
    sums = set()
    # We need to check if any two sums divide each other.
    # And if all sums are distinct.
    
    # Generate all subset sums
    # Since n is small (<= 30), 2^n is manageable if we are careful?
    # But we want to use the fast check during construction.
    # This _validate is for final check or initial seed.
    
    num_subsets = 1 << n
    for mask in range(1, num_subsets):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
        
    sums_list = sorted(sums)
    # Fast check: 
    # For each u, check if any v in sums_list is a multiple of u.
    # Using the divisor/multiple logic is better for incremental, 
    # but for a static set, we can do:
    # Sort sums.
    # For each u, check multiples?
    # Or for each v, check divisors?
    # Checking divisors of v and seeing if they are in the set is O(V * sqrt(V)) or O(V * tau(V)).
    # V is max sum.
    
    for v in sums_list:
        # Check if any proper divisor of v is in sums
        # We only need to check divisors d such that d < v.
        # And d must be in sums.
        # Iterate divisors of v.
        for i in range(1, int(math.isqrt(v)) + 1):
            if v % i == 0:
                if i in sums and i != v:
                    return False
                other = v // i
                if other in sums and other != v:
                    return False
    return True

def construct():
    deadline = time.time() + 85
    best_set = [31, 32, 34]
    best_size = 3
    
    if _validate(best_set):
        best_size = len(best_set)
        
    random.seed(42)
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Start with a random large number
        start_val = random.randint(100, 256)
        current = [start_val]
        sums = {start_val}
        max_sum = start_val
        
        candidates = [x for x in range(1, 257) if x != start_val]
        # Prefer larger numbers to keep sums large (reducing multiple checks)
        # But we need distinct elements.
        # Let's sort candidates in descending order to try large ones first?
        # Or random shuffle?
        # Large numbers create large sums, which have few multiples.
        # So trying large numbers first might be faster.
        candidates.sort(reverse=True)
        
        # We want to add as many as possible.
        # Greedy: try to add each candidate.
        # But we might get stuck. Random restarts help.
        
        # To improve diversity, shuffle the candidates?
        # If we always pick the largest valid one, we might end up with a specific set.
        # Let's shuffle the candidates.
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
                
            # Generate new sums
            new_sums = set()
            valid = True
            
            # Check if x itself is a sum?
            # x is a new element. {x} is a subset.
            # If x is in sums, then x is a sum of a subset of current.
            # Then we have two subsets with sum x. Invalid.
            if x in sums:
                continue
                
            # Generate sums including x
            for s in sums:
                ns = s + x
                if ns in sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if not valid:
                continue
                
            new_sums.add(x)
            
            # Now check divisibility between new_sums and sums
            # And within new_sums.
            
            # We can check each new sum v:
            # 1. Does any divisor of v exist in sums U new_sums (excluding v)?
            # 2. Does any multiple of v exist in sums U new_sums (excluding v)?
            
            # To be efficient, we can check against the combined set?
            # But we haven't added them yet.
            # We can check:
            # For each v in new_sums:
            #   - Check divisors of v in (sums U new_sums)
            #   - Check multiples of v in (sums U new_sums)
            
            # Let's create a temporary set of all sums
            # all_sums = sums U new_sums
            # But we can just check against sums and new_sums separately.
            
            # Optimization: 
            # If we check divisors of v, we cover u | v.
            # If we check multiples of v, we cover v | u.
            
            # Let's iterate through new_sums.
            for v in new_sums:
                # Check divisors
                # We need to check if any divisor d of v (d < v) is in sums or new_sums.
                # Note: d cannot be v.
                for i in range(1, int(math.isqrt(v)) + 1):
                    if v % i == 0:
                        d1 = i
                        if d1 < v:
                            if d1 in sums or d1 in new_sums:
                                valid = False
                                break
                        d2 = v // i
                        if d2 < v and d2 != d1:
                            if d2 in sums or d2 in new_sums:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
                
                # Check multiples
                # We need to check if any multiple m of v (m > v) is in sums or new_sums.
                # m = 2v, 3v, ...
                # Max sum is max(sums) + x (approx).
                # We can just loop while m <= max_possible_sum.
                # max_possible_sum = max_sum + x
                m = 2 * v
                while m <= max_sum + x:
                    if m in sums or m in new_sums:
                        valid = False
                        break
                    m += v
                if not valid:
                    break
                    
            if valid:
                current.append(x)
                sums.update(new_sums)
                max_sum += x # Actually max sum increases by x
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    return list(best_set)
