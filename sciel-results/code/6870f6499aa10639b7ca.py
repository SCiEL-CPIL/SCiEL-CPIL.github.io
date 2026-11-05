import time
from math import factorial

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    target = factorial(12)
    
    # Get all divisors
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    divisors = sorted(divisors)
    
    # Filter > 12
    valid_divs = [d for d in divisors if d > 12]
    n_divs = len(valid_divs)
    
    # Precompute log target for potential pruning?
    # Not needed.
    
    def can_solve_with_limit(limit):
        # Find end index
        # valid_divs is sorted
        # We want valid_divs[i] <= limit
        # Use binary search or linear scan (n is small ~600)
        end_idx = 0
        for i in range(n_divs):
            if valid_divs[i] <= limit:
                end_idx = i + 1
            else:
                break
        
        if end_idx == 0:
            return None
            
        cands = valid_divs[:end_idx]
        n = end_idx
        
        def _dfs(start_idx, current_prod, current_factors):
            if time.time() - start_time > DEADLINE:
                return None
            
            if current_prod == target:
                if 1 <= len(current_factors) <= 12:
                    return list(current_factors)
                return None
            
            if current_prod > target:
                return None
                
            if len(current_factors) > 7:
                return None
            
            # Pruning:
            remaining_needed = 7 - len(current_factors)
            if remaining_needed < 0:
                return None
            
            # Max possible product with remaining_needed factors from cands[start_idx:]
            # The largest factors are at the end of cands.
            # But we must pick from start_idx onwards.
            # So the largest available are cands[n-1], cands[n-2], ...
            
            # If we pick the largest 'remaining_needed' factors from the available set,
            # what is the max product?
            # It's the product of the last 'remaining_needed' elements in cands[start_idx:].
            
            # If current_prod * that_max_prod < target, return None.
            
            # To get the largest 'remaining_needed' elements from cands[start_idx:]:
            # They are cands[n-1], cands[n-2], ..., cands[n-remaining_needed]
            # But only if n-remaining_needed >= start_idx.
            # If not, we can't pick that many.
            
            if n - start_idx < remaining_needed:
                 # Can't pick enough factors?
                 # Actually, we don't HAVE to pick 7 factors.
                 # We can pick fewer.
                 # So this pruning is invalid if we allow k < 7.
                 # But we established k <= 7.
                 # If we can't pick 7, we can pick fewer.
                 # So we should check if we can reach target with *any* number of factors up to 7.
                 
                 # The bound "max product" assumes we pick the max number of factors.
                 # If we pick fewer, the product is smaller.
                 # So if the product of the LARGEST possible set (size 7) is < target,
                 # then any smaller set will also be < target.
                 # So the bound holds.
                 
                 # But if n - start_idx < remaining_needed, we can't pick 7.
                 # We can pick at most n - start_idx.
                 # So max_prod is product of all remaining?
                 # No, we can pick any subset.
                 # The max product is achieved by picking the largest available factors.
                 # If we can pick at most K factors, we pick the largest K.
                 
                 max_pickable = min(remaining_needed, n - start_idx)
                 if max_pickable == 0:
                     return None
                     
                 # Product of the largest max_pickable elements in cands[start_idx:]
                 # These are cands[n-1], cands[n-2], ... cands[n-max_pickable]
                 
                 max_prod = 1
                 for j in range(max_pickable):
                     max_prod *= cands[n - 1 - j]
                     
                 if current_prod * max_prod < target:
                     return None

            for i in range(start_idx, n):
                d = cands[i]
                
                if d > target // current_prod:
                    break
                    
                if time.time() - start_time > DEADLINE:
                    return None

                res = _dfs(i + 1, current_prod * d, current_factors + [d])
                if res:
                    return res
                    
            return None
        
        return _dfs(0, 1, [])

    # Binary search for the minimum M
    lo = 13
    hi = target
    
    # Check if solution exists for hi
    # It always does (target itself).
    
    while lo < hi:
        mid = (lo + hi) // 2
        sol = can_solve_with_limit(mid)
        if sol:
            hi = mid
        else:
            lo = mid + 1
            
    final_sol = can_solve_with_limit(lo)
    
    if final_sol:
        return final_sol
    else:
        return [target]
