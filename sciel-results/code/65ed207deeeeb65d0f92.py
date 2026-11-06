import sys
from math import factorial, log
from functools import lru_cache
import time
import bisect

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Get all divisors of target
    all_divs = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
        i += 1

    # Only divisors > 12, sorted
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)

    if n_divs == 0:
        return [target]

    # We want to minimize the largest factor.
    # Strategy: binary search on the maximum allowed factor value M.
    # For a given M, check if we can decompose target into distinct divisors > 12, all <= M.

    def check_solution(max_val):
        """
        Checks if target can be decomposed into distinct integers from valid_divs,
        all <= max_val, with product exactly target.
        Returns list of factors if found, else None.
        """
        # Get candidates: divisors <= max_val and > 12
        # valid_divs is sorted, so we can find the upper bound
        idx = bisect.bisect_right(valid_divs, max_val)
        candidates = valid_divs[:idx]
        n = len(candidates)

        if n == 0:
            return None

        # Memoization for DFS
        # State: (index, remaining_product)
        # We track the set of used factors implicitly by only moving forward in index (ensures distinctness)
        
        @lru_cache(maxsize=None)
        def dfs(i, rem):
            if rem == 1:
                return True
            if i >= n:
                return False
            
            # Pruning 1: If remaining product is 1, we are done (handled above)
            # Pruning 2: If rem is prime and not in candidates[i:], fail
            # Pruning 3: Upper bound check - can we form rem using at most (12 - used_count) factors?
            # Actually, we don't track count easily in state (i, rem) without it.
            # But since factors are > 12, the maximum number of factors is limited by log(12!) / log(13).
            # 12! ~ 4.7e8. 13^6 ~ 4.8e6, 13^7 ~ 6e7, 13^8 ~ 8e8. So at most 7-8 factors.
            # Let's add a count parameter to be safe and enable better pruning.
            # But lru_cache with (i, rem, count) might have too many states?
            # rem is a divisor of target, so number of possible rems is number of divisors ~ few thousand.
            # i is up to n_divs.
            # Let's stick to (i, rem) and rely on the fact that if rem > 1, we must pick more factors.
            
            c = candidates[i]
            
            # Option 1: Skip candidates[i]
            if dfs(i + 1, rem):
                return True
            
            # Option 2: Take candidates[i] if it divides rem
            if rem % c == 0:
                if dfs(i + 1, rem // c):
                    return True
            
            return False

        if dfs(0, target):
            # Reconstruct solution
            result = []
            i, rem = 0, target
            while rem != 1:
                if i >= n:
                    break
                c = candidates[i]
                # Did we skip or take?
                # Check if skipping leads to solution
                if dfs(i + 1, rem):
                    i += 1
                else:
                    # Must have taken c
                    # Check if rem % c == 0 (should be)
                    if rem % c != 0:
                        break # Should not happen
                    result.append(c)
                    rem //= c
                    i += 1
            
            return result
        else:
            return None

    # Binary search for the minimal maximum factor
    # The maximum factor must be at least target (trivial solution) and at least the smallest divisor > 12.
    # Actually, the maximum factor in the optimal solution is some divisor of target.
    # We can binary search over the sorted valid_divs to find the minimal valid_divs[mid] such that
    # check_solution(valid_divs[mid]) is not None.
    
    low = 0
    high = n_divs - 1
    best_result = None
    
    # We know [target] is a valid solution, so high = n_divs-1 (which is target) should work.
    # Let's verify check_solution(target) works quickly.
    # If it's too slow, we might need a different approach.
    # However, the state space of (i, rem) is manageable.
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        max_val = valid_divs[mid]
        
        sol = check_solution(max_val)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1

    if best_result is not None:
        return best_result
    
    # Fallback: trivial solution
    return [target]

# EVOLVE-BLOCK-END
