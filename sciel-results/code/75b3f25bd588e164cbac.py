import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin for 60s limit
    
    target = factorial(12)
    
    # Generate all divisors of target that are > 12
    divisors = []
    # Iterate up to sqrt(target) to find divisors
    i = 1
    while i * i <= target:
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
        i += 1
    divisors.sort()
    
    # Map for quick lookup
    divisor_set = set(divisors)
    
    best_max = target # Initial worst case: single factor
    best_factors = [target]
    
    def dfs(remaining, last_factor, current_factors):
        nonlocal best_max, best_factors
        
        # Check time
        if time.time() > deadline:
            return False # Signal timeout
        
        if remaining == 1:
            if len(current_factors) > 0:
                current_max = current_factors[-1] # Since sorted
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return True # Found a valid solution
            
        if len(current_factors) >= 12:
            return False
            
        # Pruning: 
        # 1. The next factor must be > last_factor
        # 2. The next factor must be < best_max (to improve)
        # 3. If we pick one more factor, it must be 'remaining'. 
        #    So if len(current_factors) == 11, the next factor is fixed.
        
        # Lower bound for next factor:
        min_next = last_factor + 1
        
        # If we can pick exactly one more factor to finish:
        # The factor must be 'remaining'.
        # It must be > last_factor and > 12 and < best_max.
        
        # Iterate through possible next factors
        # We can iterate through divisors of 'remaining' that are > last_factor
        # and < best_max (if we want to improve)
        
        # Optimization: 
        # If we have picked k factors, we have up to 12-k factors left.
        # The product of the remaining factors is 'remaining'.
        # The smallest possible product for m distinct factors > last_factor is 
        # (last_factor+1)*(last_factor+2)*...
        # If 'remaining' is too small to be formed by the required number of factors, prune.
        # But we don't know the exact number of factors left, it's variable 1..(12-len(current)).
        
        # Let's just iterate through divisors of 'remaining'
        # To speed up, we can precompute divisors for each possible 'remaining'? 
        # No, 'remaining' changes. But 'remaining' is always a divisor of target.
        # We can find divisors of 'remaining' by checking against our global 'divisors' list?
        # Or just iterate i from min_next to sqrt(remaining)?
        
        # Since target is ~4.8e8, sqrt is ~22000.
        # The number of divisors is manageable.
        
        # Let's find divisors of 'remaining' on the fly, filtered by > last_factor and < best_max
        # Actually, we can just iterate through the global 'divisors' list?
        # No, 'remaining' is smaller.
        
        # Efficient way: iterate i from min_next to int(sqrt(remaining))
        # If remaining % i == 0:
        #   candidate1 = i
        #   candidate2 = remaining // i
        
        # We need to check if candidate is > last_factor and < best_max (to improve)
        # And candidate must be > 12.
        
        limit = int(remaining**0.5) + 1
        for i in range(min_next, limit):
            if time.time() > deadline:
                return False
            if remaining % i == 0:
                # Candidate 1
                if i > 12 and i < best_max:
                    current_factors.append(i)
                    dfs(remaining // i, i, current_factors)
                    current_factors.pop()
                    
                # Candidate 2
                other = remaining // i
                if other != i:
                    if other > 12 and other < best_max and other > i: # other > i is implied by i < sqrt
                        current_factors.append(other)
                        dfs(remaining // other, other, current_factors)
                        current_factors.pop()
        
        # What if remaining itself is a candidate? 
        # This is covered if i=1, but we start from min_next > 12.
        # If remaining is a prime > last_factor, it won't be found in the loop if min_next > sqrt(remaining).
        # So we must check 'remaining' itself as a single factor to be picked.
        # This happens if we pick 'remaining' as the next (and final) factor.
        
        if remaining > 12 and remaining > last_factor and remaining < best_max:
             # Check if we can add it. 
             # We need to ensure it's a divisor (it is, remaining/remaining=1)
             current_factors.append(remaining)
             dfs(1, remaining, current_factors)
             current_factors.pop()

    dfs(target, 12, [])
    
    return best_factors
