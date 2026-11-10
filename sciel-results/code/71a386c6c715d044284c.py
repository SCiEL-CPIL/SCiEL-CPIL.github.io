import time
from math import factorial
from functools import lru_cache
from itertools import combinations

TARGET = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Prime factorization of 12!
    # 12! = 479001600
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors of TARGET
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    divisors.sort()
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # For each valid divisor, compute its prime exponent vector
    def prime_exp_vec(n):
        vec = [0] * 5
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Map divisor to its prime exponent vector
    divisor_to_vec = {}
    for d in valid_divisors:
        divisor_to_vec[d] = prime_exp_vec(d)
    
    # Target exponent vector
    target_vec = tuple(exps)
    
    # Memoization: cache (remaining_vec, last_factor, used_set_frozenset) -> best_max_found_or_None
    # But used_set is expensive. Instead, we'll use a different approach.
    
    # Better approach: search for the minimum possible max factor by trying candidate max values
    # For a given max M, check if we can find distinct factors > 12, all <= M, with product = TARGET
    
    def can_factor_with_max(M, time_limit):
        """Check if TARGET can be factored into distinct integers > 12, all <= M, with product = TARGET."""
        # Filter valid divisors that are <= M
        candidates = [d for d in valid_divisors if d <= M]
        if not candidates:
            return None
        
        # We need to find a subset of candidates with product = TARGET
        # Use DFS with pruning
        
        best_solution = None
        
        def dfs(remaining, last_factor, current_factors, depth):
            nonlocal best_solution
            
            if time.time() > time_limit:
                return False
            
            if remaining == 1:
                if len(current_factors) >= 1:
                    best_solution = list(current_factors)
                return True
            
            # If remaining is prime and > last_factor, it must be the last factor
            # Check if remaining is prime
            is_prime = True
            for p in primes:
                if p * p > remaining:
                    break
                if remaining % p == 0:
                    is_prime = False
                    break
            if is_prime:
                if remaining > last_factor and remaining <= M and remaining > 12:
                    current_factors.append(remaining)
                    best_solution = list(current_factors)
                    current_factors.pop()
                    return True
                return False
            
            # Iterate through possible next factors
            for d in candidates:
                if d <= last_factor:
                    continue
                if d > remaining:
                    break
                if remaining % d != 0:
                    continue
                
                rem2 = remaining // d
                
                # Pruning: can rem2 be factored into distinct integers > d, all <= M?
                if rem2 > 1:
                    # rem2 must have at least one divisor > d and <= M
                    if rem2 < d + 1:
                        continue
                    # Also, the smallest product of k distinct integers > d is (d+1)(d+2)...
                    # For k=1: need rem2 > d
                    # For k>=2: need rem2 >= (d+1)*(d+2)
                    # But we don't know k, so just check rem2 > d for now
                    if rem2 <= d:
                        continue
                
                current_factors.append(d)
                found = dfs(rem2, d, current_factors, depth + 1)
                current_factors.pop()
                
                if found:
                    return True
                
                # If we've tried enough, break
                if depth > 10:
                    break
            
            return False
        
        dfs(TARGET, 12, [], 0)
        return best_solution
    
    # Binary search on the maximum factor
    # Lower bound: at least 13 (smallest factor > 12)
    # Upper bound: TARGET (single factor)
    
    lo = 13
    hi = TARGET
    best_solution = [TARGET]
    
    # First, try to find any solution to establish a good upper bound
    # Use the original DFS to find a solution quickly
    def quick_find_solution():
        nonlocal deadline
        best = [TARGET]
        
        def dfs(remaining, last_factor, current_factors, depth):
            nonlocal best
            if time.time() > deadline:
                return
            if remaining == 1:
                if len(current_factors) >= 1:
                    if max(current_factors) < max(best):
                        best = list(current_factors)
                return
            
            for d in valid_divisors:
                if d <= last_factor:
                    continue
                if d > remaining:
                    break
                if remaining % d != 0:
                    continue
                
                rem2 = remaining // d
                if rem2 > 1 and rem2 <= d:
                    continue
                
                current_factors.append(d)
                dfs(rem2, d, current_factors, depth + 1)
                current_factors.pop()
                
                if d >= max(best):
                    break
        
        dfs(TARGET, 12, [], 0)
        return best
    
    # Find an initial solution
    initial_solution = quick_find_solution()
    if initial_solution and len(initial_solution) > 1:
        hi = max(initial_solution)
        best_solution = initial_solution
    else:
        hi = TARGET
    
    # Binary search for the minimum possible max factor
    # We want to find the smallest M such that can_factor_with_max(M) returns a valid solution
    # But we also need to ensure the solution has all factors <= M
    
    # Since binary search on M might be expensive, let's try a different approach:
    # Search over possible numbers of factors (1 to 12) and find the solution that minimizes the max
    
    # For each number of factors k, find the solution with k factors that minimizes the max
    # This is still hard, but we can use the binary search on M for each k
    
    # Actually, let's just do a direct search with better pruning
    # The key insight: we want to minimize the max, so we should try to keep all factors small
    
    # Let's use a priority queue approach: try solutions in order of their max factor
    # But that's complex. Instead, let's improve the DFS with stronger pruning.
    
    # Improved DFS with memoization on (remaining, last_factor)
    # Since the set of used factors matters, we can't easily memoize, but we can use
    # the fact that we're searching for the minimum max to prune more aggressively.
    
    def optimized_dfs():
        nonlocal deadline
        best_max = TARGET
        best_sol = [TARGET]
        
        # Memoize: (remaining, last_factor) -> min_max_factor_achievable_from_here
        # This is an approximation, but it can help prune
        memo = {}
        
        def dfs(remaining, last_factor, current_factors, depth):
            nonlocal best_max, best_sol
            
            if time.time() > deadline:
                return
            
            if remaining == 1:
                if len(current_factors) >= 1:
                    max_f = max(current_factors)
                    if max_f < best_max:
                        best_max = max_f
                        best_sol = list(current_factors)
                return
            
            # Check memo
            key = (remaining, last_factor)
            if key in memo:
                if memo[key] <= last_factor:  # Can't do better than what we've already tried
                    return
            
            # Try to find a solution
            found_better = False
            for d in valid_divisors:
                if d <= last_factor:
                    continue
                if d > remaining:
                    break
                if d >= best_max:
                    break  # Can't improve
                if remaining % d != 0:
                    continue
                
                rem2 = remaining // d
                if rem2 > 1 and rem2 <= d:
                    continue
                
                current_factors.append(d)
                dfs(rem2, d, current_factors, depth + 1)
                current_factors.pop()
                
                if best_max <= d:
                    break
            
            memo[key] = last_factor
        
        dfs(TARGET, 12, [], 0)
        return best_sol, best_max
    
    result, _ = optimized_dfs()
    if result and len(result) >= 1:
        return result
    else:
        return [TARGET]
