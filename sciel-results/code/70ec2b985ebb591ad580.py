"""Solve: factor 12! into 1..12 DISTINCT integers > 12, minimize the LARGEST factor."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def _get_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i > 12:
                divs.append(i)
            if i != n // i and n // i > 12:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    deadline = time.time() + 85  # Safety margin under 90s kill
    target = factorial(12)
    all_divs = _get_divisors(target)
    n_divs = len(all_divs)
    
    # We want to find the smallest M such that there exists a subset of divs
    # where all d <= M, d > 12, distinct, and product is target.
    # Binary search on the index in all_divs, or the value.
    # Let's binary search on the value M.
    # Lower bound: smallest divisor > 12 is 13.
    # Upper bound: target.
    
    lo = 13
    hi = target
    best_sol = [target] # Fallback
    best_max = target
    
    # Helper to check if a solution exists with max factor <= M
    def can_solve_with_max(M):
        # Filter divisors <= M
        # Since all_divs is sorted, we can find the cutoff index
        # Using bisect for efficiency if we did this many times, but binary search on M
        # means we do O(log(target)) calls.
        # n_divs is ~800. Filtering is fast.
        valid_divs = [d for d in all_divs if d <= M]
        if not valid_divs:
            return False, []
        
        # DFS
        # We need to pick distinct divisors from valid_divs that multiply to target.
        # Sort valid_divs (already sorted).
        # Pruning:
        # 1. If remaining == 1: success
        # 2. If remaining < valid_divs[idx]: fail (smallest available is too big)
        # 3. If we have picked 12 factors: fail (if remaining != 1)
        
        def dfs(remaining, idx, count, current_factors):
            if time.time() > deadline:
                return False, []
            
            if remaining == 1:
                return True, current_factors
            
            if count >= 12:
                return False, []
            
            if idx >= len(valid_divs):
                return False, []
            
            # If the smallest available divisor is greater than remaining, we can't form remaining
            # (since all factors are integers > 1, product of any subset will be >= smallest factor)
            # Wait, if remaining is 1, handled. If remaining > 1, we need factors >= 13.
            # If valid_divs[idx] > remaining, then no divisor in valid_divs[idx:] can divide remaining
            # because they are all > remaining.
            if valid_divs[idx] > remaining:
                return False, []
                
            # Additional prune: If remaining is prime, it must be in valid_divs
            # and we must pick it.
            # Check if remaining is prime
            # is_prime check is cheap for numbers < 10^9
            def is_prime(n):
                if n < 2: return False
                if n == 2: return True
                if n % 2 == 0: return False
                for i in range(3, isqrt(n) + 1, 2):
                    if n % i == 0:
                        return False
                return True
            
            if is_prime(remaining):
                # It must be in valid_divs and we must pick it
                # Find if remaining is in valid_divs[idx:]
                # Since valid_divs is sorted, we can bisect
                import bisect
                pos = bisect.bisect_left(valid_divs, remaining, idx)
                if pos < len(valid_divs) and valid_divs[pos] == remaining:
                    current_factors.append(remaining)
                    # If we picked it, remaining becomes 1, which is success
                    # But we must ensure distinctness. 
                    # Since we pick in increasing order of indices, and we only pick each index once,
                    # distinctness is guaranteed by the index constraint.
                    # However, we must ensure we don't pick the same VALUE if it appears twice?
                    # Divisors are unique values.
                    return True, current_factors
                else:
                    return False, []
            
            for i in range(idx, len(valid_divs)):
                d = valid_divs[i]
                if d > remaining:
                    break
                if remaining % d != 0:
                    continue
                
                current_factors.append(d)
                found, sol = dfs(remaining // d, i + 1, count + 1, current_factors)
                current_factors.pop()
                
                if found:
                    return True, sol
                    
            return False, []
        
        return dfs(target, 0, 0, [])

    # Binary search for the minimal M
    # We want to find the smallest M such that can_solve_with_max(M) is True.
    # Note: The function returns (True, solution) or (False, []).
    
    # Optimization: We can just iterate M from the list of divisors?
    # The optimal M must be one of the divisors.
    # So we can binary search on the index of all_divs.
    
    left = 0
    right = n_divs - 1
    final_sol = None
    
    # Ensure we have a valid range. 
    # all_divs[0] is 13. all_divs[-1] is 12!.
    # 12! is always a valid solution (single factor).
    
    while left <= right:
        if time.time() > deadline:
            break
        mid = (left + right) // 2
        M = all_divs[mid]
        success, sol = can_solve_with_max(M)
        if success:
            final_sol = sol
            best_max = M
            best_sol = sol
            right = mid - 1
        else:
            left = mid + 1
            
    if final_sol:
        return final_sol
    else:
        return best_sol

# EVOLVE-BLOCK-END
