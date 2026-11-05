import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, keep 10s buffer
    
    target = factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    divisors = sorted(divisors)
    
    # We want to minimize the largest factor.
    # We iterate through possible maximum factors M.
    # M must be one of the divisors.
    
    def can_form(rem, start_idx, max_idx, current_factors):
        if rem == 1:
            return current_factors[:]
        
        # Pruning: if product of all remaining available factors is less than rem, return None
        # This is expensive to compute every time. 
        # Alternative: just rely on the loop.
        
        # Heuristic: try larger divisors first? 
        # If we try larger divisors first, we might find a solution faster 
        # because we reduce 'rem' faster?
        # Or try smaller? 
        # Let's try from end to start (largest to smallest) within the allowed range.
        # This might help find a valid combination quickly.
        
        for i in range(max_idx, start_idx - 1, -1):
            d = divisors[i]
            if rem % d == 0:
                # Check time
                if time.time() > deadline:
                    return None
                
                res = can_form(rem // d, i + 1, max_idx, current_factors + [d])
                if res is not None:
                    return res
        
        return None

    # Iterate M from smallest to largest
    # M is divisors[j]
    for j in range(len(divisors)):
        M = divisors[j]
        # The available divisors are divisors[0] to divisors[j]
        # We need to check if target can be formed by a subset of divisors[0..j]
        
        # Optimization: The product of all available divisors must be >= target.
        # Actually, we just need a subset.
        
        # Check if solution exists with max factor <= M
        # The search space is divisors[0..j]
        
        # We call can_form(target, 0, j, [])
        # Note: can_form expects start_idx and max_idx.
        # My can_form logic: 
        # i goes from max_idx down to start_idx.
        # If we pick divisors[i], we recurse with start_idx = i+1.
        # This ensures we only pick indices > i.
        # So we are picking a subset of indices from [start_idx, max_idx].
        # Wait, if i goes down to start_idx, and we pick i, next start is i+1.
        # This means we pick indices in increasing order?
        # If we pick i, we cannot pick any index <= i.
        # So we are picking a set of indices $i_1 < i_2 < \dots$
        # In the loop, we iterate $i$ from $max\_idx$ down to $start\_idx$.
        # If we pick $i$, we recurse with $start\_idx = i+1$.
        # The next loop will iterate from $max\_idx$ down to $i+1$.
        # So we pick indices in strictly increasing order of their position in the list?
        # No, we pick $i$, then from the remaining indices $> i$, we pick $k$.
        # So the indices picked are $i < k < \dots$.
        # This covers all subsets of $[start\_idx, max\_idx]$.
        
        res = can_form(target, 0, j, [])
        
        if res is not None:
            # We found a decomposition with max factor <= M.
            # Since we iterate M from small to large, this is the minimal M.
            # The factors in res are distinct and > 12.
            # We should return them.
            # The order doesn't matter for the product, but let's sort them for consistency.
            return sorted(res)
        
        # Optional: Check if time is running out
        if time.time() > deadline:
            # Fallback: return the seed solution or best found so far?
            # The seed is [12!].
            # If we haven't found a better one, return [12!].
            # But [12!] is a valid solution (1 factor, distinct, >12).
            # The score is the largest factor.
            # So returning [12!] is always a valid fallback.
            return [target]
            
    # If loop finishes without finding (should not happen), return [target]
    return [target]
