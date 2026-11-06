import math
from functools import lru_cache

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    # We want to find the smallest M such that a valid subset exists with max <= M
    # Iterate M from 13 upwards
    # To speed up, we can start from a better lower bound
    # Lower bound for max factor:
    # If we use k factors, max >= (target)^(1/k)
    # Also max >= 13
    # Let's just start from 13. The search space is small.
    
    best_solution = None
    
    # We can iterate M from 13 to target
    # But we can break early if we find a solution.
    
    # Optimization: Instead of iterating M and filtering candidates each time,
    # we can just use the full list of valid_divisors and add a constraint in DFS:
    # d <= current_max_bound.
    # But iterating M and checking is simple.
    
    # Let's refine the DFS to take a max_limit.
    
    def dfs(rem, start_idx, max_limit, current_factors):
        if rem == 1:
            return True
        if start_idx >= len(valid_divisors):
            return False
        
        # Pruning: if the smallest available divisor is greater than rem, we can't multiply to rem
        if valid_divisors[start_idx] > rem:
            return False
            
        # Pruning: if rem is prime
        # A prime number can only be factored into 1 * prime.
        # Since we are picking factors > 12, if rem is prime, rem itself must be one of the factors.
        # So rem must be in valid_divisors[start_idx:] and rem <= max_limit.
        # If rem is prime, we must pick rem.
        # Check if rem is prime.
        # For speed, we can just try to pick rem if it's in the list.
        # But checking primality for every rem is expensive.
        # However, rem decreases quickly.
        
        for i in range(start_idx, len(valid_divisors)):
            d = valid_divisors[i]
            if d > max_limit:
                break
            if d > rem:
                break
            if rem % d == 0:
                # Pick d
                current_factors.append(d)
                if dfs(rem // d, i + 1, max_limit, current_factors):
                    return True
                current_factors.pop()
                
        return False

    # Iterate max_limit
    # Lower bound:
    # The minimum possible max factor is at least 13.
    # Let's start from 13.
    
    # To be safe and fast, we can estimate a lower bound.
    # (12!)^(1/12) ~ 40.
    # So starting from 13 is fine, it will be fast.
    
    for M in range(13, target + 1):
        # Optimization: if M is small, the list of candidates is small.
        # We can pass M to dfs.
        factors = []
        if dfs(target, 0, M, factors):
            return factors
    
    # Fallback (should not happen)
    return [target]
