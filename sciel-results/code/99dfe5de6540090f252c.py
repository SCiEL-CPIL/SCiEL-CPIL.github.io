import time
from math import factorial

def construct():
    target = factorial(12)
    deadline = time.time() + 85
    
    # Pre-compute divisors? Or just check on the fly.
    # We iterate M.
    
    def can_form(target, candidates):
        """
        candidates: sorted list of distinct integers > 12 that divide target.
        Returns True if a subset of candidates has product target.
        """
        n = len(candidates)
        # Precompute suffix products for pruning?
        # suffix_prod[i] = product of candidates[i:]
        # If suffix_prod[i] < target_remaining, prune.
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        def dfs(rem, start):
            if time.time() > deadline:
                return None
            if rem == 1:
                return []
            if start >= n:
                return None
            # Pruning: even if we take all remaining, can we reach rem?
            if suffix_prod[start] < rem:
                return None
            
            for i in range(start, n):
                c = candidates[i]
                if c > rem:
                    break
                if rem % c != 0:
                    continue
                
                # Recurse
                res = dfs(rem // c, i + 1)
                if res is not None:
                    return [c] + res
            return None
            
        return dfs(target, 0)

    # Iterate M
    for M in range(13, 1000): # Upper bound can be higher, but 1000 is safe
        if time.time() > deadline:
            # Fallback: return a valid solution if found earlier? 
            # We haven't found one yet in this loop structure.
            # We should probably store the best valid solution found in a previous M?
            # No, if M is small, we didn't find a solution.
            # If we run out of time, we should return the smallest M for which we found a solution?
            # Actually, if we run out of time, we might not have found any.
            # But we can always return [target] if we haven't found better?
            # Wait, [target] is a valid solution (1 factor, distinct, > 12).
            # But we want to minimize the largest factor.
            # If we time out, we should return the best found so far.
            # Since we iterate M upwards, the first one we find is the optimal.
            # If we don't find one, we return [target].
            pass
        
        # Build candidates
        # Only divisors of target
        candidates = []
        for x in range(13, M + 1):
            if target % x == 0:
                candidates.append(x)
        
        if not candidates:
            continue
            
        # Check if solution exists
        sol = can_form(target, candidates)
        if sol is not None:
            # We found a solution with max factor M
            # Since we iterate M upwards, this is the minimal max factor
            return sol
            
    # If loop finishes without finding a solution (should not happen, [target] is always possible if M >= target)
    # But we iterate M up to 1000. target is 4.7e8.
    # We should ensure M goes high enough or handle the [target] case.
    # Actually, if M < target, we might not find a solution.
    # The single factor [target] has max factor target.
    # So we should start with best = [target] and update if we find better.
    
    # Revised strategy:
    best_factors = [target]
    best_max = target
    
    for M in range(13, best_max + 1):
        if time.time() > deadline:
            break
            
        candidates = [x for x in range(13, M + 1) if target % x == 0]
        if not candidates:
            continue
        
        sol = can_form(target, candidates)
        if sol is not None:
            # This solution has max factor <= M
            # Since we iterate M, the first time we find a solution for a given M,
            # it implies the max factor is exactly M (or less, but we would have found it earlier).
            # Actually, if a solution exists with max factor M', it will be found when M=M'.
            # So the first M where can_form returns True is the optimal max factor.
            return sol
            
    return best_factors
