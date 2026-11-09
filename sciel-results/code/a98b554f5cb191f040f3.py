"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
import sys

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _greedy_complete(remaining: Fraction, used: set, start_d: int, deadline: float):
    """
    Complete the Egyptian fraction greedily from the current state.
    Returns a list of denominators added, or None if it fails (exceeds limits).
    """
    result = []
    r = remaining
    last_d = start_d
    while r > 0:
        if time.time() > deadline:
            return None
        if r.numerator == 0:
            break
        # Standard greedy step: ceil(1/r)
        d = (r.denominator + r.numerator - 1) // r.numerator
        
        # Ensure distinctness and increasing order
        while d in used or d <= last_d:
            d += 1
            if d > MAX_DENOM:
                return None
            
        if d > MAX_DENOM:
            return None
        
        result.append(d)
        used.add(d)
        r -= Fraction(1, d)
        last_d = d
        
        if len(result) > MAX_TERMS:
            return None
    
    return result

def _try_solution(denoms):
    """
    Validate a list of denominators and return the largest one if valid, else None.
    """
    if not denoms:
        return None
    if len(denoms) > MAX_TERMS:
        return None
    if len(set(denoms)) != len(denoms):
        return None
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return None
    
    total = sum(Fraction(1, d) for d in denoms)
    if total != TARGET:
        return None
    
    return max(denoms)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known high-quality solution: [8, 31, 248] -> max 248
    # Let's verify it quickly
    known_sol = [8, 31, 248]
    best_max = _try_solution(known_sol)
    if best_max is None:
        # Fallback to greedy if hardcoded fails (should not happen)
        r = TARGET
        denoms = []
        used = set()
        while r > 0 and time.time() < deadline:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used or (denoms and d <= denoms[-1]):
                d += 1
            denoms.append(d)
            used.add(d)
            r -= Fraction(1, d)
            if len(denoms) > MAX_TERMS:
                break
        if r == 0:
            best_max = _try_solution(denoms)
    
    if best_max is None:
        return known_sol # Return best effort even if validation had a bug, though it shouldn't

    # Search for better solutions (max < best_max)
    # Strategy: Try different first denominators
    # If we pick first denominator d1, remaining is 5/31 - 1/d1 = (5*d1 - 31) / (31*d1)
    # We need to decompose this remainder.
    # To minimize the MAX denominator, we generally want the denominators to be as small as possible,
    # but the constraints of distinct integers and exact sum make this a search problem.
    
    # A common heuristic: try denominators d1 such that d1 is a divisor of something useful or simply iterate.
    # Since we want to minimize the MAX, let's try to find a solution with max < best_max.
    
    # Let's bound the search for d1.
    # If max_denom < best_max, then d1 < best_max.
    # Also d1 must be such that the remainder is positive: 1/d1 < 5/31 => d1 > 31/5 = 6.2 => d1 >= 7.
    
    # We can iterate d1 from 7 to best_max - 1.
    # For each d1, we have a remainder R.
    # We can then try to complete R using a limited search or greedy.
    # To be efficient, we can use a recursive search with pruning for the next few terms.
    
    def _search_remaining(rem, used, min_d, depth, current_max, limit_max):
        """
        Recursive search for Egyptian fraction of 'rem' with denominators > min_d,
        not in 'used', and all < limit_max.
        Returns list of denoms or None.
        """
        if time.time() > deadline:
            return None
        if rem == 0:
            return []
        if depth >= MAX_TERMS:
            return None
        if rem < 0:
            return None
        
        # Upper bound for next denominator:
        # We need 1/d < rem => d > 1/rem
        # Also we want to keep potential max below current best (or limit_max)
        
        lower_d = min_d
        # A standard bound for Egyptian fraction search:
        # If we have n terms left, the smallest possible sum is 1/n + 1/(n+1) ... which is small.
        # But a stronger bound: if we pick d, remaining is rem - 1/d.
        # We must be able to represent rem - 1/d with (MAX_TERMS - depth) terms.
        
        # Let's just iterate d from min_d to limit_max - 1
        # Optimization: 
        # We know that if we have k terms left, the max denominator cannot exceed 
        # something related to the product of previous denominators, but that's loose.
        
        # Simple iteration:
        for d in range(lower_d, limit_max):
            if d in used:
                continue
            if d > limit_max - 1:
                break
            if rem - Fraction(1, d) < 0:
                break # Since d is increasing, 1/d is decreasing, so rem - 1/d is increasing. 
                       # Wait, if rem - 1/d < 0, then 1/d > rem. 
                       # As d increases, 1/d decreases, so rem - 1/d increases.
                       # So if it's negative, it will become positive later? 
                       # No, if 1/d > rem, then d < 1/rem.
                       # So we should start from d = floor(1/rem) + 1.
            
            # Actually, let's refine the start of the loop.
            # We need 1/d < rem for the remainder to be positive? 
            # No, we can have multiple terms summing to rem. 
            # But for the *first* term of the remainder, we usually pick 1/d <= rem to keep remainder positive?
            # Not necessarily. But if 1/d > rem, the remainder is negative, which is bad for standard Egyptian (positive terms).
            # So yes, we need 1/d <= rem => d >= ceil(1/rem).
            
            pass # This pass is just to think. The loop below will handle it.
            
        # Re-implementing the loop properly inside the function
        min_d_effective = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        # If min_d_effective > limit_max - 1, no solution
        if min_d_effective >= limit_max:
            return None
            
        for d in range(min_d_effective, limit_max):
            if d in used:
                continue
            rem_new = rem - Fraction(1, d)
            if rem_new < 0:
                break
            # Pruning: 
            # If rem_new > 0, we have (MAX_TERMS - depth - 1) terms left.
            # The maximum possible sum with k terms of distinct integers > d is bounded.
            # Specifically, the largest term in the remaining solution must be < limit_max.
            # A simple check: if rem_new is still large, we might need small denominators.
            
            # Let's just recurse.
            res = _search_remaining(rem_new, used | {d}, d + 1, depth + 1, max(current_max, d), limit_max)
            if res is not None:
                return [d] + res
        return None

    # We want to find a solution with max < best_max.
    # So limit_max = best_max - 1.
    # We iterate over the first denominator d1.
    # d1 must be < best_max.
    # d1 must be >= 7.
    
    # To speed up, we can iterate d1 in a smart way.
    # Maybe iterate d1 from 7 to best_max - 1.
    
    for d1 in range(7, best_max):
        if time.time() > deadline:
            break
        if d1 in known_sol:
            # We already have a solution with d1 in known_sol, but maybe with different other terms?
            # It's worth checking if a *better* solution exists involving d1.
            # But if d1=8, we already have [8, 31, 248] with max 248.
            # If we find [8, x, y] with max < 248, that would be better.
            pass
        
        rem1 = TARGET - Fraction(1, d1)
        if rem1 <= 0:
            continue
        
        # Try to complete rem1 with max denominator < best_max
        # limit_max for the search is best_max - 1
        # The first denominator is d1, so max_so_far = d1
        # We call search for rem1 with min_d = d1 + 1, limit_max = best_max - 1
        # Wait, the search function returns the list of denoms for the REMAINDER.
        # So the total solution is [d1] + res.
        # The max of the total solution is max(d1, max(res)).
        # We require max(res) < best_max.
        
        # Optimization: If d1 >= best_max, skip (already handled by range).
        
        res = _search_remaining(rem1, {d1}, d1 + 1, 1, d1, best_max)
        
        if res is not None:
            candidate = [d1] + res
            cand_max = _try_solution(candidate)
            if cand_max is not None and cand_max < best_max:
                best_max = cand_max
                # We can continue searching for even better, or break if we hit a theoretical lower bound?
                # Theoretically, max cannot be less than something?
                # For 5/31, min max is likely small.
                # Let's just keep searching until deadline.

    return [8, 31, 248] if best_max == 248 else None # This logic is flawed. We should return the best found.
    
    # Let's refactor to track the actual best list.
    
# Refactoring construct to actually return the best found list.

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    known_sol = [8, 31, 248]
    best_max = _try_solution(known_sol)
    best_list = known_sol
    
    if best_max is None:
        return known_sol

    # Search for better
    def _search_remaining(rem, used, min_d, depth, limit_max):
        if time.time() > deadline:
            return None
        if rem == 0:
            return []
        if depth >= MAX_TERMS:
            return None
        
        min_d_eff = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if min_d_eff >= limit_max:
            return None
            
        for d in range(min_d_eff, limit_max):
            if d in used:
                continue
            rem_new = rem - Fraction(1, d)
            if rem_new < 0:
                break
            res = _search_remaining(rem_new, used | {d}, d + 1, depth + 1, limit_max)
            if res is not None:
                return [d] + res
        return None

    # Iterate d1
    for d1 in range(7, best_max):
        if time.time() > deadline:
            break
        rem1 = TARGET - Fraction(1, d1)
        if rem1 <= 0:
            continue
        res = _search_remaining(rem1, {d1}, d1 + 1, 1, best_max)
        if res is not None:
            candidate = [d1] + res
            cand_max = _try_solution(candidate)
            if cand_max is not None and cand_max < best_max:
                best_max = cand_max
                best_list = candidate
                
    return best_list

# EVOLVE-BLOCK-END
