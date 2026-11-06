import sys
from math import factorial
from functools import lru_cache
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
        
    def check_solution(max_idx):
        """
        Checks if a solution exists using valid_divs[0..max_idx].
        Returns the list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        n = len(candidates)
        
        # Precompute suffix products for pruning
        # suffix_prod[i] = product(candidates[i:])
        # If rem > suffix_prod[i], it's impossible to form rem with remaining candidates.
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        # Also, if rem is not 1, we need at least 1 factor.
        # If rem < candidates[i], we can't use candidates[i] or any larger one?
        # No, we can skip. But if we need to form rem, and all remaining candidates are > rem, 
        # we can only use them if rem is 1 (which is handled) or if we don't need to use them?
        # If rem > 1 and all remaining candidates are > rem, we can't form rem.
        # So if candidates[i] > rem and rem != 1, we can prune? 
        # Not exactly, we might skip candidates[i] and find a smaller one? No, sorted ascending.
        # So if candidates[i] > rem, then all candidates[j] for j>=i are > rem.
        # We can only form rem if rem==1. If rem!=1, we fail.
        
        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= 12:
                return False
            
            # Pruning 1: If the product of all remaining candidates is less than rem, fail.
            if rem > suffix_prod[i]:
                return False
                
            # Pruning 2: If the smallest remaining candidate is greater than rem, 
            # we cannot form rem (since we need factors <= rem to multiply to rem, 
            # unless rem is 1 which is handled). 
            # Wait, if we have rem=15 and candidates are [20, 30], we can't form 15.
            # If we have rem=15 and candidates are [15, 20], we can pick 15.
            # So if candidates[i] > rem, we can't pick any candidate to divide rem.
            # We must skip all? If we skip all, rem remains rem != 1, and i becomes n.
            # So if candidates[i] > rem, we return False immediately?
            # Yes, because any factor used must be a divisor of rem, so factor <= rem.
            # Since candidates are sorted, if the first one is > rem, all are > rem.
            if candidates[i] > rem:
                return False

            c = candidates[i]
            
            # Heuristic: Try taking c first if it divides rem.
            # This encourages using more factors, which is good for minimizing the max factor
            # (since we are checking feasibility under a cap M, finding a dense solution is good).
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
            while rem != 1:
                if i >= n:
                    break
                c = candidates[i]
                
                # Check if we can solve by taking i
                # Note: We must mimic the decision logic of dfs exactly.
                # dfs tries Take first if divisible, then Skip.
                
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
                # Verify that skipping leads to a solution (it should, if dfs returned True)
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
    best_result = None
    
    # Fallback: if binary search fails or times out, return [target]
    # Note: [target] is always a valid solution (1 factor, distinct, >12).
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        sol = check_solution(mid)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1
    
    if best_result is not None:
        return best_result
    
    # If no solution found in time or via search, return the trivial solution
    # The trivial solution is [12!]
    return [target]

# EVOLVE-BLOCK-END
