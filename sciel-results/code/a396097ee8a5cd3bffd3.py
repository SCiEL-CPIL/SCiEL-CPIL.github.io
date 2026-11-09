import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Generate all divisors of target that are > 12
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    if target > 12:
        divisors.append(target)
    
    divisors = sorted(set(divisors))
    n_div = len(divisors)
    
    # We want to find a subset of distinct divisors (all > 12) that multiply to target,
    # minimizing the largest element.
    # Since divisors are sorted, we can binary search on the maximum index allowed.
    
    # Memoization for feasibility checks.
    # Key: (rem, start_idx)
    # Value: The smallest limit_idx for which a solution exists from this state,
    #        or a large number if infeasible for the current context (but we handle this carefully).
    # Actually, simpler approach:
    # We perform a DFS that tries to find a solution.
    # To optimize, we use a global memo that stores the result of (rem, start_idx) -> bool/None
    # but the 'limit_idx' constraint makes this tricky.
    
    # Alternative strategy:
    # Since the number of divisors is ~792, and the product is fixed,
    # the branching factor is very small.
    # Let's try a direct DFS with strong pruning, iterating the max allowed value.
    
    # But binary search on max index is better.
    # Let's implement a function can_form(rem, start_idx, limit_idx)
    # with memoization keyed by (rem, start_idx, limit_idx)? No, limit_idx changes.
    
    # Key insight:
    # If a solution exists using elements from start_idx to limit_idx,
    # then it also exists using elements from start_idx to limit_idx' where limit_idx' > limit_idx.
    # So the property is monotonic with respect to limit_idx.
    
    # We can use a memo that stores the MINIMUM limit_idx required for a solution from (rem, start_idx).
    # If we know that min_limit[rem, start_idx] = L, then:
    # - If current limit_idx >= L, return True.
    # - If current limit_idx < L, return False.
    # This is powerful!
    
    # Let's implement this.
    # min_limit[(rem, start_idx)] = the smallest index k such that a solution exists using divisors[start_idx..k].
    # If no solution exists, min_limit is infinity.
    
    INF = float('inf')
    min_limit = {}
    
    def get_min_limit(rem, start_idx, current_limit_idx):
        """
        Returns the minimum index k >= start_idx such that a solution exists
        using divisors[start_idx..k] that multiplies to 'rem'.
        If no such k exists <= current_limit_idx, returns INF.
        Actually, we want to know if a solution exists within [start_idx, current_limit_idx].
        We can compute the true minimum required index, and compare it to current_limit_idx.
        """
        if rem == 1:
            return start_idx  # No more factors needed, so min index is start_idx (effectively 0 extra)
        
        key = (rem, start_idx)
        if key in min_limit:
            # We have computed the minimum index required for this state.
            # If the stored value is <= current_limit_idx, then it's feasible.
            # But wait, the stored value is the GLOBAL minimum index.
            # If global_min <= current_limit_idx, then yes.
            return min_limit[key]
        
        # Pruning
        if start_idx >= n_div:
            min_limit[key] = INF
            return INF
        
        # If the smallest available divisor is > rem, we can't divide.
        if divisors[start_idx] > rem:
            min_limit[key] = INF
            return INF
        
        # We need to pick a divisor d from divisors[start_idx..current_limit_idx]
        # such that rem % d == 0 and the remaining product rem//d can be formed from divisors[i+1..current_limit_idx].
        
        best = INF
        for i in range(start_idx, current_limit_idx + 1):
            d = divisors[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            # Check if rem // d can be formed from divisors[i+1..current_limit_idx]
            sub_min = get_min_limit(rem // d, i + 1, current_limit_idx)
            
            # If sub_min is not INF, then a solution exists.
            # The maximum index used in this branch is max(i, sub_min).
            # Actually, the constraint is that all indices must be <= current_limit_idx.
            # Since sub_min is the minimum index required for the subproblem,
            # and we are using index i for the current factor,
            # the total maximum index used is max(i, sub_min).
            # But we are checking if it fits within current_limit_idx.
            # Since sub_min is computed with the same current_limit_idx constraint?
            # No, get_min_limit(rem//d, i+1, current_limit_idx) returns the minimum index k such that
            # a solution exists using divisors[i+1..k] with k <= current_limit_idx.
            # If sub_min <= current_limit_idx, then it's valid.
            # The maximum index used in the full solution is max(i, sub_min).
            # We want to minimize the maximum index overall.
            # So for this branch, the max index is max(i, sub_min).
            
            if sub_min != INF:
                max_idx_used = max(i, sub_min)
                if max_idx_used < best:
                    best = max_idx_used
        
        min_limit[key] = best
        return best
    
    # Now, we binary search on the limit_idx.
    # We want the smallest limit_idx such that get_min_limit(target, 0, limit_idx) <= limit_idx.
    # Actually, get_min_limit(target, 0, limit_idx) returns the minimum max-index used.
    # If that value is <= limit_idx, then a solution exists within the limit.
    
    low = 0
    high = n_div - 1
    best_solution = None
    best_limit = -1
    
    # We need to reconstruct the solution.
    # The memoization above only stores the min index, not the path.
    # We can do a second pass to reconstruct, using the same logic.
    
    def reconstruct(rem, start_idx, limit_idx):
        if rem == 1:
            return []
        
        if start_idx >= n_div:
            return None
        
        for i in range(start_idx, limit_idx + 1):
            d = divisors[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            # Check if subproblem is feasible
            sub_min = get_min_limit(rem // d, i + 1, limit_idx)
            if sub_min != INF and sub_min <= limit_idx:
                sub_sol = reconstruct(rem // d, i + 1, limit_idx)
                if sub_sol is not None:
                    return [d] + sub_sol
        
        return None
    
    # Binary search
    for _ in range(20):
        if time.time() > deadline:
            break
        
        mid = (low + high) // 2
        if mid < 0:
            break
        
        min_idx_needed = get_min_limit(target, 0, mid)
        
        if min_idx_needed <= mid:
            # Solution exists within limit mid
            best_limit = mid
            high = mid - 1
        else:
            low = mid + 1
        
        if low > high:
            break
    
    # Find the best solution
    if best_limit != -1:
        # Reconstruct with the best limit found
        # We might need to clear the memoization or ensure it's consistent.
        # The memoization is valid for any limit_idx >= the stored min index.
        # Since we are using best_limit, and get_min_limit was called with various limits,
        # the memo might contain entries computed with smaller limits.
        # However, get_min_limit(rem, start, limit) is monotonic: if it returns L for limit1,
        # it will return <= L for limit2 > limit1.
        # So the cached values are valid lower bounds.
        # To be safe, we can clear the memo and re-run for the final limit.
        
        min_limit.clear()
        sol = reconstruct(target, 0, best_limit)
        if sol is not None:
            return sol
    
    # Fallback
    return [target]
