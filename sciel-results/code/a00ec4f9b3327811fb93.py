import math
import bisect
from functools import lru_cache

def construct():
    N = 12
    target = math.factorial(N)
    
    # Get all divisors of target
    divisors = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
        i += 1
        
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    if not valid_divisors:
        return [target]
        
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def find_solution_with_max(M):
        idx = bisect.bisect_right(valid_divisors, M)
        candidates = valid_divisors[:idx]
        
        if not candidates:
            return None
            
        solution = []
        
        def dfs(rem, start_idx):
            if rem == 1:
                return True
            if start_idx >= len(candidates):
                return False
            
            # Pruning: 
            # If rem is prime, it must be in candidates[start_idx:]
            if is_prime(rem):
                # Find if rem is in candidates[start_idx:]
                # Since candidates is sorted, we can use bisect
                pos = bisect.bisect_left(candidates, rem, start_idx)
                if pos < len(candidates) and candidates[pos] == rem:
                    solution.append(rem)
                    if dfs(1, pos): # pos is index of rem, next must be < pos
                        return True
                    solution.pop()
                return False
                
            # Iterate from end to start_idx
            # We want to pick large factors first to reduce rem quickly
            for i in range(len(candidates) - 1, start_idx - 1, -1):
                c = candidates[i]
                if c > rem:
                    continue
                if rem % c == 0:
                    solution.append(c)
                    if dfs(rem // c, i):
                        return True
                    solution.pop()
            return False
        
        if dfs(target, 0):
            return solution
        return None

    # Binary search on the index of valid_divisors
    lo_idx = 0
    hi_idx = len(valid_divisors) - 1
    best_sol = None
    
    while lo_idx <= hi_idx:
        mid_idx = (lo_idx + hi_idx) // 2
        M = valid_divisors[mid_idx]
        sol = find_solution_with_max(M)
        if sol is not None:
            best_sol = sol
            hi_idx = mid_idx - 1
        else:
            lo_idx = mid_idx + 1
            
    if best_sol is None:
        return [target]
        
    return best_sol
