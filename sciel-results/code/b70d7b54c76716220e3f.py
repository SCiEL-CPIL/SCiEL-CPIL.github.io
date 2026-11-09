import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for 60s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    candidates = sorted([d for d in divisors if d > 12])
    
    # Precompute: for each candidate, check if it's a divisor of target
    # (already guaranteed by construction)
    
    # Strategy: Find the minimum M such that target can be written as product
    # of distinct factors from {d in candidates : d <= M}, with at most 12 factors.
    
    # Use DFS with memoization on (rem, start_idx) but with the constraint
    # that we track the max factor used.
    
    # Better: binary search on M, and for each M, check if decomposition exists.
    # But we can do better: single DFS that minimizes max.
    
    # Let's use a different approach: 
    # Factorize target into primes, then search for groupings.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We need to group these prime powers into distinct integers > 12,
    # minimizing the largest.
    
    # Prime factorization of 12!:
    # 2: floor(12/2)+floor(12/4)+floor(12/8) = 6+3+1 = 10
    # 3: floor(12/3)+floor(12/9) = 4+1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # 11: floor(12/11) = 1
    
    # So target = 2^10 * 3^5 * 5^2 * 7 * 11 = 1024 * 243 * 25 * 7 * 11
    # = 479001600 ✓
    
    # The number of divisors is (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # That's manageable.
    
    # For each possible M (from min candidate upward), check if we can decompose.
    # Use DFS with strong pruning.
    
    best_solution = None
    best_max = float('inf')
    
    # Sort candidates in ascending order
    # We'll iterate M from smallest to largest
    
    # Optimization: precompute for each divisor d > 12, whether d divides target
    # (all do by construction)
    
    # DFS: try to factor 'rem' using distinct candidates from index 'start_idx' onward,
    # all <= max_val. Return the list of factors if successful.
    
    def can_decompose(rem, start_idx, max_val, path, time_limit):
        if time.time() > time_limit:
            return None
        
        if rem == 1:
            return path[:]
        
        if len(path) >= 12:  # Max 12 factors
            return None
        
        # If rem is prime and rem > max_val, we can't use it (it's > max_val)
        # and we can't factor it further with numbers <= max_val
        if rem > max_val:
            # Check if rem can be factored into two or more factors all <= max_val
            # If rem is prime, impossible
            if _is_prime(rem):
                return None
            # Otherwise, we need to find a factor f of rem with 12 < f <= max_val
            # and rem/f can also be decomposed
        
        if start_idx >= len(candidates):
            return None
        
        # Pruning: if rem < 13, impossible (all factors must be > 12)
        if rem < 13:
            return None
        
        # Find the smallest candidate >= start_idx that is <= max_val and divides rem
        # If none exists, fail
        
        found_any = False
        for i in range(start_idx, len(candidates)):
            f = candidates[i]
            if f > max_val:
                break
            if f > rem:
                break
            if rem % f != 0:
                continue
            
            found_any = True
            new_rem = rem // f
            new_path = path + [f]
            result = can_decompose(new_rem, i + 1, max_val, new_path, time_limit)
            if result is not None:
                return result
        
        return None
    
    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Iterate M from smallest candidate upward
    for idx, M in enumerate(candidates):
        if time.time() > deadline:
            break
        
        sol = can_decompose(target, 0, M, [], deadline)
        if sol is not None:
            best_solution = sol
            break  # First valid M is the smallest possible max
    
    if best_solution is None:
        # Fallback: single factor
        return [target]
    
    return best_solution
