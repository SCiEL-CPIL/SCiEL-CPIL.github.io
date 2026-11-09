from functools import lru_cache
import math

def construct():
    target = math.factorial(12)
    
    # Generate divisors > 12
    divs = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            if i > 12:
                divs.append(i)
            if i != target // i and target // i > 12:
                divs.append(target // i)
    divs = sorted(set(divs))
    
    n = len(divs)
    
    # Binary search on the max index
    # low: 0, high: n-1
    # We want the smallest max_idx such that a solution exists.
    
    @lru_cache(maxsize=None)
    def dfs(start_idx, remaining, count, max_idx):
        if remaining == 1:
            return 1 <= count <= 12
        
        if count >= 12:
            return False
            
        if start_idx > max_idx:
            return False
            
        # Pruning: 
        # If the product of all remaining divisors from start_idx to max_idx is < remaining, 
        # it's impossible. But calculating product is expensive.
        # Simpler: If remaining is not divisible by any d in divs[start_idx:max_idx+1], return False.
        # This is checked in the loop.
        
        for i in range(start_idx, max_idx + 1):
            d = divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                if dfs(i + 1, remaining // d, count + 1, max_idx):
                    return True
        return False

    # Binary search
    lo, hi = 0, n - 1
    best_idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        dfs.cache_clear() # Important to clear cache for new max_idx? 
                         # Actually, the state depends on max_idx. 
                         # If we change max_idx, the results for the same (start, rem, count) 
                         # might change. So we must clear or include max_idx in key.
                         # Including max_idx in key is safer.
                         # Let's include max_idx in the function signature or key.
                         # My current dfs takes max_idx as arg, so lru_cache will treat it as part of the key.
                         # So we don't need to clear cache, but the cache will grow.
                         # However, since max_idx changes, old entries are useless.
                         # It's better to clear or use a new function.
                         # Given small n, clearing is fine.
        
        # Wait, if I clear cache, I lose memoization across binary search steps?
        # No, each step is independent.
        
        if dfs(0, target, 0, mid):
            best_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_idx == -1:
        return [target] # Should not happen
        
    # Reconstruct solution
    # We need to find the actual factors.
    # We can run a similar DFS that returns the path.
    
    def find_path(start_idx, remaining, count, max_idx):
        if remaining == 1:
            if 1 <= count <= 12:
                return []
            else:
                return None
        
        if count >= 12:
            return None
            
        if start_idx > max_idx:
            return None
            
        for i in range(start_idx, max_idx + 1):
            d = divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                res = find_path(i + 1, remaining // d, count + 1, max_idx)
                if res is not None:
                    return [d] + res
        return None

    path = find_path(0, target, 0, best_idx)
    if path:
        return path
    else:
        return [target]
