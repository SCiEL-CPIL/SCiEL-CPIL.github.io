import time
from math import factorial

def construct():
    target = factorial(12)
    deadline = time.time() + 85.0
    
    # Get all divisors of target greater than 12
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    # Ensure target itself is considered if it were a divisor (it is), but we only iterate up to sqrt.
    # The loop above adds pairs. 
    # We need to make sure we don't miss anything.
    # Actually, iterating i from 13 to sqrt(target) and adding i and target//i covers all divisors > 12.
    # Except if target itself is a divisor (it is) and target > sqrt(target), it is added as the 'other' when i=1? No, i starts at 13.
    # If i=1, 1 is not > 12.
    # If i=2, 2 is not > 12.
    # ...
    # If i=13, we add 13 and target//13.
    # This covers all divisors d where d > 12 and d <= sqrt(target), and their complements.
    # What about divisors d > sqrt(target)? They are the complements of d' < sqrt(target).
    # If d' > 12, then d' was added as 'i' or 'other'.
    # If d' <= 12, then d is not added by the loop starting at 13.
    # Example: divisor 14400. Complement is 33280. 
    # If 14400 > 12, it is added.
    # If complement 33280 > 12, it is added.
    # What if divisor is 1? Not > 12.
    # What if divisor is 2? Not > 12.
    # So any divisor d > 12 either has a complement d' > 12 (both added) or d' <= 12.
    # If d' <= 12, then d > target/12.
    # The loop only adds d if i (the smaller factor) > 12.
    # So if d' <= 12, d is NOT added by the loop.
    # We must add divisors d > 12 whose complement is <= 12.
    
    divisors_set = set(divisors)
    # Add divisors d > 12 where target//d <= 12
    # This means d >= target // 12
    min_d_for_small_comp = target // 12 + 1
    for d in range(min_d_for_small_comp, target + 1):
        if target % d == 0:
            divisors_set.add(d)
            
    divisors = sorted(list(divisors_set))
    
    # To speed up, we can pre-filter divisors that are obviously too large? 
    # No, we iterate M upwards.
    
    # We want to find the smallest M such that there is a subset of divisors <= M 
    # that are distinct and product to target.
    
    # We iterate M over the divisors.
    # If a solution exists with max factor M, it will be found when we test M.
    
    # DFS to check if target can be formed by distinct divisors from divisors[:end_idx]
    # end_idx is the index such that divisors[end_idx] > M
    
    def can_form(rem, start_idx, end_idx):
        if rem == 1:
            return []
        
        if start_idx >= end_idx:
            return None
            
        # Pruning: if the smallest available divisor > rem, we can't form rem
        if divisors[start_idx] > rem:
            return None
            
        # Pruning: if rem is prime, it must be in the list
        # Simple check: if rem is in divisors[start_idx:end_idx]
        # But we need to return the list of factors.
        
        for i in range(start_idx, end_idx):
            d = divisors[i]
            if d > rem:
                break
            if rem % d == 0:
                sub_sol = can_form(rem // d, i + 1, end_idx)
                if sub_sol is not None:
                    return [d] + sub_sol
        return None

    # Iterate M
    # We can optimize by not testing every M, but the number of divisors is small (~700).
    # Testing 700 cases with a fast DFS should be well within 85s.
    
    for i, M in enumerate(divisors):
        if time.time() > deadline:
            break
            
        # The available divisors are divisors[0 ... i]
        # We need to form 'target' using distinct divisors from this set.
        
        sol = can_form(target, 0, i + 1)
        if sol is not None:
            return sol
            
    # Fallback: return a single factor (target) if no better solution found
    # This is valid: 1 factor, distinct, > 12, product is target.
    # Max factor is target.
    return [target]
