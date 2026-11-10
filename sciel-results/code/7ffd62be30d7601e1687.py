import time
from math import factorial
from functools import lru_cache

def _factorize(n):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    target = factorial(n)
    
    # Generate all divisors of 12! that are > 12
    target_factors = _factorize(target)
    primes = sorted(target_factors.keys())
    max_exps = [target_factors[p] for p in primes]
    
    # Generate all divisors > 12
    divisors = []
    def gen_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > n:
                divisors.append(current_val)
            return
        p = primes[idx]
        val = current_val
        for e in range(max_exps[idx] + 1):
            gen_divisors(idx + 1, val)
            val *= p
            if val > target:
                break
                
    gen_divisors(0, 1)
    divisors = sorted(set(divisors))
    n_divs = len(divisors)
    
    # We want to find the smallest index k such that there exists a subset of 
    # divisors[0..k] (distinct) whose product is target.
    # The largest factor in such a subset will be at most divisors[k].
    # To minimize the largest factor, we want the smallest such k.
    # Since the property is monotonic, we can binary search on k.
    
    # However, we also need to return the actual factors.
    # We can do a DFS that returns the factors if a solution is found.
    
    # Optimization: Precompute which divisors divide which? 
    # Not strictly necessary given the small size.
    
    # Let's define the DFS for a fixed limit index k.
    # We use a dictionary for memoization of (rem, start) -> (success, path) or just success
    # But we need the path. Returning paths in memo can be memory heavy if not careful, 
    # but depth is <= 12, so paths are small.
    
    def find_solution(limit_idx):
        """
        Finds a set of distinct divisors from divisors[0..limit_idx] 
        whose product is target.
        Returns the list of factors if found, else None.
        """
        memo = {}
        
        def dfs(rem, start):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return []
            
            if start > limit_idx:
                return None
                
            key = (rem, start)
            if key in memo:
                return memo[key]
            
            # Pruning: 
            # If the product of the largest possible remaining divisors cannot reach rem?
            # Or if the smallest divisor at start is greater than rem?
            # Since divisors are sorted, if divisors[start] > rem, we can't use any 
            # divisor from start onwards that is > rem. But we might have skipped some?
            # No, we are iterating from start. If divisors[start] > rem, then all 
            # divisors[start..limit_idx] are > rem (since sorted). 
            # So we can't pick any. Return None.
            if divisors[start] > rem:
                memo[key] = None
                return None
            
            # Another pruning: 
            # The maximum product we can get from remaining divisors is 
            # divisors[limit_idx] * divisors[limit_idx-1] * ...
            # But we are limited by rem.
            
            for i in range(start, limit_idx + 1):
                d = divisors[i]
                if d > rem:
                    # Since sorted, no larger divisor will divide rem
                    # (unless rem is 0, which it isn't)
                    # Actually, if d > rem, then d cannot divide rem.
                    # And all subsequent divisors are >= d > rem.
                    # So we can break.
                    break
                
                if rem % d == 0:
                    res = dfs(rem // d, i + 1)
                    if res is not None:
                        memo[key] = [d] + res
                        return memo[key]
            
            memo[key] = None
            return None
        
        return dfs(target, 0)
    
    # Binary search for the smallest limit_idx
    lo = 0
    hi = n_divs - 1
    best_sol = None
    best_limit = n_divs
    
    # We can just iterate from 0 upwards? 
    # n_divs is 792. 
    # DFS is very fast. 
    # Iterating 0..791 and running DFS for each might be 792 * (DFS cost).
    # DFS cost is roughly exponential in depth but pruned heavily.
    # Binary search is safer for time.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        sol = find_solution(mid)
        if sol is not None:
            best_sol = sol
            best_limit = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_sol is not None:
        return best_sol
        
    # Fallback: return the single factor 12!
    # This is always a valid solution (1 factor, distinct, > 12, product 12!)
    # But it's not optimal. If the search fails, we return this.
    return [target]

# EVOLVE-BLOCK-END
