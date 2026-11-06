"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    target = factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 75
    
    def get_divisors(n):
        divisors = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
            i += 1
        return sorted(divisors)
    
    all_divisors = get_divisors(target)
    
    def find_decomposition(max_factor):
        candidates = [d for d in all_divisors if 13 <= d <= max_factor]
        if not candidates:
            return None
        
        result = []
        
        def dfs(idx, remaining, depth):
            if time.time() > deadline:
                return False
            if remaining == 1:
                return bool(result)
            if idx >= len(candidates) or depth >= 12:
                return False
            if remaining < candidates[idx]:
                return False
            
            if remaining % candidates[idx] == 0:
                result.append(candidates[idx])
                if dfs(idx + 1, remaining // candidates[idx], depth + 1):
                    return True
                result.pop()
            
            if dfs(idx + 1, remaining, depth):
                return True
            
            return False
        
        if dfs(0, target, 0):
            return list(result)
        return None
    
    lo, hi = 13, target
    best = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        res = find_decomposition(mid)
        if res is not None:
            best = res
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best:
        return best
    return [target]
# EVOLVE-BLOCK-END
