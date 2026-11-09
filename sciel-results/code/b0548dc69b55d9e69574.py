import time
from math import factorial, isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target that are > 12
    divisors_set = set()
    for i in range(1, isqrt(target) + 1):
        if target % i == 0:
            if i > 12:
                divisors_set.add(i)
            other = target // i
            if other != i and other > 12:
                divisors_set.add(other)
    
    candidates = sorted(divisors_set)
    n_cand = len(candidates)
    
    # For fast lookup: check if a value is a valid candidate
    cand_set = set(candidates)
    
    # Binary search on the maximum allowed value
    # We want the smallest M such that a solution exists with all factors <= M
    
    def can_form(rem, start_idx, limit_idx, max_factors):
        """
        Can we form 'rem' using distinct candidates from candidates[start_idx..limit_idx],
        using at most max_factors factors?
        Returns True if yes, False if no.
        """
        if rem == 1:
            return True
        
        if start_idx > limit_idx:
            return False
        
        if max_factors <= 0:
            return False
        
        # Pruning: if rem is greater than the product of the largest available candidates,
        # we can't form it. But computing that product is expensive.
        # Simpler pruning: if the smallest available candidate > rem, we can't form rem
        # unless rem itself is a candidate (but we're picking from start_idx onwards,
        # and rem might be smaller than candidates[start_idx])
        if candidates[start_idx] > rem:
            return False
        
        # Pruning: if rem is prime, it must be in candidates[start_idx..limit_idx]
        # Check if rem is prime
        if rem > 1:
            is_prime = True
            for p in range(2, isqrt(rem) + 1):
                if rem % p == 0:
                    is_prime = False
                    break
            if is_prime:
                # rem must be in candidates[start_idx..limit_idx]
                # Binary search for rem in candidates
                pos = bisect_right(candidates, rem) - 1
                if pos >= start_idx and pos <= limit_idx and candidates[pos] == rem:
                    return True
                return False
        
        # Try picking each candidate from start_idx to limit_idx
        for i in range(start_idx, limit_idx + 1):
            d = candidates[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            # Check if rem // d can be formed from candidates[i+1..limit_idx]
            if can_form(rem // d, i + 1, limit_idx, max_factors - 1):
                return True
        
        return False
    
    # Binary search on the index of the maximum allowed candidate
    low = 0
    high = n_cand - 1
    best_idx = -1
    
    # We need at least 1 factor, so the minimum max factor is at least candidates[0]
    # But we need to check if a single factor works (target itself)
    
    # First, check if target is in candidates (it should be, since target > 12)
    # If target is in candidates, then [target] is a valid solution with max = target
    
    # Binary search for the smallest index such that can_form(target, 0, idx, 12) is True
    for _ in range(20):  # At most 20 iterations for binary search
        if time.time() > deadline:
            break
        
        if low > high:
            break
        
        mid = (low + high) // 2
        
        if can_form(target, 0, mid, 12):
            best_idx = mid
            high = mid - 1
        else:
            low = mid + 1
    
    # Reconstruct the solution
    def reconstruct(rem, start_idx, limit_idx, max_factors):
        if rem == 1:
            return []
        
        if start_idx > limit_idx:
            return None
        
        if max_factors <= 0:
            return None
        
        # Pruning: if rem is prime
        if rem > 1:
            is_prime = True
            for p in range(2, isqrt(rem) + 1):
                if rem % p == 0:
                    is_prime = False
                    break
            if is_prime:
                pos = bisect_right(candidates, rem) - 1
                if pos >= start_idx and pos <= limit_idx and candidates[pos] == rem:
                    return [rem]
                return None
        
        for i in range(start_idx, limit_idx + 1):
            d = candidates[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            sub = reconstruct(rem // d, i + 1, limit_idx, max_factors - 1)
            if sub is not None:
                return [d] + sub
        
        return None
    
    if best_idx != -1:
        sol = reconstruct(target, 0, best_idx, 12)
        if sol is not None and len(sol) >= 1 and len(sol) <= 12:
            # Verify: all distinct, all > 12, product = target
            prod = 1
            for f in sol:
                prod *= f
            if prod == target and len(set(sol)) == len(sol) and all(f > 12 for f in sol):
                return sol
    
    # Fallback: return [target]
    return [target]
