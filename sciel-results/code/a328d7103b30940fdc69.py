import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 75  # safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Global best
    best_max = N
    best_factors = [N]
    
    # Precompute all divisors of N that are > 12
    divisors_gt12 = []
    for i in range(13, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors_gt12.append(i)
            other = N // i
            if other > 12 and other != i:
                divisors_gt12.append(other)
    divisors_gt12 = sorted(set(divisors_gt12))
    
    # Build a set for O(1) lookup
    divisor_set = set(divisors_gt12)
    
    # For each divisor d of N, precompute divisors of N//d that are > 12
    # This will help with the DFS
    # Actually, let's compute divisors on the fly but with better bounds
    
    def get_divisors_of(m, min_val=13):
        """Get all divisors of m that are >= min_val, sorted."""
        if m == 1:
            return []
        result = []
        for i in range(min_val, int(math.isqrt(m)) + 1):
            if m % i == 0:
                result.append(i)
                other = m // i
                if other >= min_val and other != i:
                    result.append(other)
        if m >= min_val:
            result.append(m)
        return sorted(set(result))
    
    best_found = [N]  # mutable holder for best solution
    
    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max, best_factors
        
        if time.time() - start_time > deadline:
            return False
        
        if depth == target_k:
            if current_prod == N:
                max_f = path[-1]  # factors are in increasing order
                if max_f < best_max:
                    best_max = max_f
                    best_factors = list(path)
            return True  # explored this branch
        
        remaining = target_k - depth
        
        # Check if current_prod divides N
        if N % current_prod != 0:
            return False
        
        target_rem = N // current_prod
        
        # Minimum product of 'remaining' distinct integers > last_factor:
        # (last_factor+1) * (last_factor+2) * ... * (last_factor+remaining)
        min_prod = 1
        for i in range(1, remaining + 1):
            min_prod *= (last_factor + i)
        
        if target_rem < min_prod:
            return False
        
        # Maximum product of 'remaining' distinct integers where each is at most some bound
        # For pruning: if we're trying to achieve max factor < best_max,
        # then all remaining factors must be < best_max
        # So max possible product is (best_max-1) * (best_max-2) * ... * (best_max-remaining)
        # But this is only useful if best_max is already found
        if best_max < N:
            max_prod = 1
            for i in range(remaining):
                max_prod *= (best_max - 1 - i)
                if max_prod > target_rem:
                    break
            if target_rem > max_prod:
                return False
        
        # If remaining == 1, the factor is determined
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12:
                # Check if it's valid (it will be, since x > last_factor > 12)
                max_f = x
                if max_f < best_max:
                    best_max = max_f
                    best_factors = path + [x]
            return True
        
        # For the next factor x:
        # x > last_factor, x > 12
        # x divides target_rem
        # The remaining (remaining-1) factors must be > x, distinct, and their product = target_rem / x
        # Also, x * (x+1) * ... * (x + remaining - 1) <= target_rem
        
        lower_bound = max(last_factor + 1, 13)
        
        # Upper bound: x * (x+1) * ... * (x + remaining - 1) <= target_rem
        # Find max x such that this holds
        x_upper = lower_bound
        while True:
            prod_test = 1
            for i in range(remaining):
                prod_test *= (x_upper + i)
                if prod_test > target_rem:
                    break
            if prod_test > target_rem:
                break
            x_upper += 1
            if x_upper > 100000:  # safety
                break
        x_upper -= 1
        
        # Also bound by best_max - 1 if we already have a solution
        if best_max < N:
            x_upper = min(x_upper, best_max - 1)
        
        if x_upper < lower_bound:
            return True
        
        # Iterate x from lower_bound to x_upper, checking if x divides target_rem
        for x in range(lower_bound, x_upper + 1):
            if time.time() - start_time > deadline:
                return False
            
            if target_rem % x != 0:
                continue
            
            # Additional prune: after choosing x, the remaining (remaining-1) factors
            # must be > x and their product must be target_rem // x
            # Min product of (remaining-1) factors > x:
            # (x+1) * (x+2) * ... * (x + remaining - 1)
            min_rem_prod = 1
            for i in range(1, remaining):
                min_rem_prod *= (x + i)
            
            if target_rem // x < min_rem_prod:
                continue
            
            # Max product if we want to beat best_max:
            # All remaining factors must be < best_max
            # So they must be in range (x+1) to (best_max-1)
            # Max product: (best_max-1) * (best_max-2) * ... * (best_max-(remaining-1))
            if best_max < N:
                max_rem_prod = 1
                for i in range(remaining - 1):
                    max_rem_prod *= (best_max - 1 - i)
                    if max_rem_prod > target_rem // x:
                        break
                if target_rem // x > max_rem_prod:
                    continue
            
            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()
        
        return True
    
    # Try different values of k. Start with k that gives smallest geometric mean.
    # N^(1/k) for k=1..12:
    # k=7: ~14.5 (best candidate for small max)
    # k=6: ~28
    # k=5: ~54
    # etc.
    
    for k in range(7, 0, -1):
        if time.time() - start_time > deadline:
            break
        dfs(k, 1, 12, 0, [])
        if best_max <= 15:
            # Very good solution found, no need to try smaller k
            break
    
    return best_factors
