import time
from math import factorial, isqrt
from functools import lru_cache

def construct():
    start_time = time.time()
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Precompute all divisors of target
    divisors = set()
    limit = isqrt(target)
    for i in range(1, limit + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: only divisors > 12, sorted
    valid_divisors = sorted([d for d in divisors if d > 12])
    n_divs = len(valid_divisors)
    
    # Map divisor to index for quick lookup
    div_idx = {d: i for i, d in enumerate(valid_divisors)}
    
    # Precompute: for each divisor d, which larger divisors divide it
    # Actually, we need: for a given remaining value R and last factor L,
    # what are the valid next factors?
    
    # is_prime helper
    def is_prime(n):
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
    
    # Memoized feasibility check
    # State: (remaining, last_factor, slots_left, max_allowed)
    # Use dict for memo since arguments vary widely
    memo = {}
    
    def can_decompose(remaining, last_factor, slots_left, max_allowed):
        if time.time() - start_time > 78:
            return False
        
        if remaining == 1:
            return True
        if slots_left == 0:
            return False
        
        key = (remaining, last_factor, slots_left, max_allowed)
        if key in memo:
            return memo[key]
        
        # If remaining is prime, must pick it
        if is_prime(remaining):
            result = (remaining > last_factor and remaining <= max_allowed and remaining > 12)
            memo[key] = result
            return result
        
        # Upper bound: each factor <= max_allowed, so remaining <= max_allowed^slots_left
        if remaining > max_allowed ** slots_left:
            memo[key] = False
            return False
        
        # Lower bound: minimum product of 'slots_left' distinct integers > last_factor
        # is (last_factor+1)*(last_factor+2)*...*(last_factor+slots_left)
        # If remaining < that, impossible
        min_prod = 1
        for k in range(1, slots_left + 1):
            min_prod *= (last_factor + k)
        if remaining < min_prod:
            memo[key] = False
            return False
        
        # Try each candidate factor d where last_factor < d <= min(max_allowed, remaining)
        # and d divides remaining
        # Find candidates in valid_divisors
        # Use bisect to find start position
        from bisect import bisect_right, bisect_left
        start = bisect_right(valid_divisors, last_factor)
        end_limit = min(max_allowed, remaining)
        end = bisect_right(valid_divisors, end_limit)
        
        for idx in range(start, end):
            d = valid_divisors[idx]
            if remaining % d != 0:
                continue
            if can_decompose(remaining // d, d, slots_left - 1, max_allowed):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    # Binary search on M (max allowed factor)
    # First establish upper bound
    hi = target
    for test_M in [100, 200, 500, 1000, 5000, 10000, 100000, 1000000]:
        if time.time() - start_time > 70:
            break
        if can_decompose(target, 12, 12, test_M):
            hi = test_M
            break
    
    lo = 13
    while lo < hi and time.time() - start_time < 75:
        mid = (lo + hi) // 2
        if can_decompose(target, 12, 12, mid):
            hi = mid
        else:
            lo = mid + 1
    
    M = lo
    
    # Find actual decomposition
    def find_decomposition(remaining, last_factor, slots_left, max_allowed):
        if remaining == 1:
            return []
        
        from bisect import bisect_right
        start = bisect_right(valid_divisors, last_factor)
        end_limit = min(max_allowed, remaining)
        end = bisect_right(valid_divisors, end_limit)
        
        for idx in range(start, end):
            d = valid_divisors[idx]
            if remaining % d != 0:
                continue
            rest = find_decomposition(remaining // d, d, slots_left - 1, max_allowed)
            if rest is not None:
                return [d] + rest
        
        return None
    
    solution = find_decomposition(target, 12, 12, M)
    if solution is not None and len(solution) <= 12:
        return solution
    
    return [target]
