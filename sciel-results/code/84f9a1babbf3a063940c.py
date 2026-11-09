import time
from math import factorial

N = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Generate divisors of N
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            if N // i != i and N // i > 12:
                divisors.append(N // i)
    
    divisors = sorted(set(divisors))
    k = len(divisors)
    
    # Precompute prefix products for fast range product queries
    prefix_prod = [1] * (k + 1)
    for i in range(k):
        prefix_prod[i + 1] = prefix_prod[i] * divisors[i]
    
    def max_product_in_range(lo, hi, m):
        """Max product of m distinct divisors from divisors[lo..hi]"""
        if m <= 0:
            return 1
        if m > hi - lo + 1:
            m = hi - lo + 1
        start_idx = hi - m + 1
        return prefix_prod[hi + 1] // prefix_prod[start_idx]

    def can_solve(j):
        # DFS to find subset of divisors[0..j] that multiplies to N
        # returns list of factors or None
        
        def dfs(rem, start_idx, current_factors):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return current_factors
            
            if len(current_factors) >= 12:
                return None
            
            if start_idx > j:
                return None
                
            # Pruning 1: smallest divisor is too big
            if rem < divisors[start_idx]:
                return None
                
            # Pruning 2: largest possible product is too small
            slots_left = 12 - len(current_factors)
            available_count = j - start_idx + 1
            
            if slots_left > 0:
                m = min(slots_left, available_count)
                max_p = max_product_in_range(start_idx, j, m)
                if rem > max_p:
                    return None
            
            for i in range(start_idx, j + 1):
                if rem % divisors[i] == 0:
                    res = dfs(rem // divisors[i], i + 1, current_factors + [divisors[i]])
                    if res is not None:
                        return res
            return None
        
        return dfs(N, 0, [])

    for j in range(k):
        if time.time() > deadline:
            break
        sol = can_solve(j)
        if sol is not None:
            return sol
            
    return [N]
