import sys
from math import factorial
from functools import lru_cache
from bisect import bisect_left
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)
    
    # Get all divisors of target
    all_divs = []
    for i in range(1, int(target ** 0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    
    # Only divisors > 12
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [target]
    
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

    def check_solution(max_idx):
        """
        Checks if a solution exists using valid_divs[0..max_idx].
        Returns the list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        n = len(candidates)
        
        # Precompute suffix products for pruning
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        # Build a set of candidates for quick lookup
        cand_set = set(candidates)
        
        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= 12:
                return False
            
            # Pruning 1: If the product of all remaining candidates is less than rem, fail.
            if rem > suffix_prod[i]:
                return False
                
            # Pruning 2: If candidates[i] > rem, we can't use any remaining candidate
            if candidates[i] > rem:
                return False

            # Pruning 3: If rem is prime, it must be in the remaining candidates
            if is_prime(rem):
                pos = bisect_left(candidates, rem, i)
                if pos < n and candidates[pos] == rem:
                    return True
                return False

            # Pruning 4: If rem is in the remaining candidates, take it directly
            pos = bisect_left(candidates, rem, i)
            if pos < n and candidates[pos] == rem:
                if count + 1 <= 12:
                    return True

            c = candidates[i]
            
            # Try taking c if it divides rem
            if rem % c == 0:
                if dfs(i + 1, rem // c, count + 1):
                    return True
            
            # Option 2: Skip candidates[i]
            if dfs(i + 1, rem, count):
                return True
            
            return False

        if dfs(0, target, 0):
            # Reconstruct solution
            result = []
            i, rem, count = 0, target, 0
            while rem != 1 and i < n:
                c = candidates[i]
                
                took = False
                if rem % c == 0:
                    if dfs(i + 1, rem // c, count + 1):
                        took = True
                        result.append(c)
                        rem //= c
                        count += 1
                        i += 1
                        continue
                
                # If not took, we must have skipped
                if dfs(i + 1, rem, count):
                    i += 1
                else:
                    # Should not happen
                    break
            
            return result
        else:
            return None

    # Binary search for the minimal M (index in valid_divs)
    low = 0
    high = n_divs - 1
    best_result = [target]
    best_max = target
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        sol = check_solution(mid)
        
        if sol is not None and len(sol) > 0:
            max_factor = max(sol)
            if max_factor < best_max:
                best_max = max_factor
                best_result = sol
            high = mid - 1
        else:
            low = mid + 1
    
    return best_result

# EVOLVE-BLOCK-END
