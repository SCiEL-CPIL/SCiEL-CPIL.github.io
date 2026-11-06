"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    target = Fraction(5, 31)
    
    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_den = 1736
    
    def check_valid(denoms):
        """Check if denominators are valid: distinct, positive, in range, sum to target"""
        if not denoms or len(denoms) > 256:
            return False
        if any(d <= 0 or d > 10000000 for d in denoms):
            return False
        if len(set(denoms)) != len(denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target
    
    def update_candidate(denoms):
        """Update best solution if candidate is better"""
        nonlocal best_solution, best_max_den
        if not check_valid(denoms):
            return
        mx = max(denoms)
        if mx < best_max_den:
            best_max_den = mx
            best_solution = denoms[:]
    
    def greedy_complete(remainder, used_set, max_denom):
        """Complete a remainder using greedy, respecting max_denom constraint"""
        if remainder == 0:
            return []
        if remainder < 0:
            return None
        
        values = []
        r = remainder
        while r > 0:
            # Find smallest d not in used_set such that 1/d <= r and d <= max_denom
            # d >= ceil(1/r)
            d_min = (r.denominator + r.numerator - 1) // r.numerator
            if d_min < 1:
                d_min = 1
            
            found = False
            # Try from d_min upward, but limit search
            for d in range(d_min, min(d_min + 200, max_denom + 1)):
                if d in used_set:
                    continue
                new_r = r - Fraction(1, d)
                if new_r < 0:
                    continue
                values.append(d)
                used_set.add(d)
                r = new_r
                found = True
                break
            
            if not found:
                return None
            if len(values) > 200:
                return None
        
        return values
    
    def try_completion_with_bound(first_denoms, max_denom):
        """Try to complete a partial solution with max denominator bound"""
        used = set(first_denoms)
        remainder = target
        for d in first_denoms:
            remainder -= Fraction(1, d)
        
        if remainder < 0:
            return None
        if remainder == 0:
            if max(first_denoms) <= max_denom:
                return list(first_denoms)
            return None
        
        completion = greedy_complete(remainder, used, max_denom)
        if completion is None:
            return None
        
        all_denoms = list(first_denoms) + completion
        if len(set(all_denoms)) != len(all_denoms):
            return None
        if max(all_denoms) > max_denom:
            return None
        
        return all_denoms
    
    # Strategy: Try to find solutions with progressively smaller max denominators
    # We know 1736 works. Let's try to find if smaller bounds work.
    
    # First, try various starting denominators and see what max we can achieve
    # For each starting denom d, the remainder is 5/31 - 1/d
    # We need to complete this with denominators > d, all <= max_denom
    
    # Try single starting denominators
    for d1 in range(7, 100):  # 1/d < 5/31 means d > 6.2
        if time.time() - start_time > time_limit:
            break
        
        rem = target - Fraction(1, d1)
        if rem <= 0:
            continue
        
        # Try to complete with various max bounds
        # Start with a reasonable bound and see if we can do better
        # The greedy completion will give us some max, let's see if we can reduce it
        
        # Use greedy to get a baseline completion
        used = {d1}
        completion = greedy_complete(rem, used.copy(), 10000000)
        if completion is not None:
            all_denoms = [d1] + completion
            if len(set(all_denoms)) == len(all_denoms) and len(all_denoms) <= 256:
                update_candidate(all_denoms)
    
    # Try pairs of starting denominators for more control
    for d1 in range(7, 60):
        if time.time() - start_time > time_limit:
            break
        
        rem1 = target - Fraction(1, d1)
        if rem1 <= 0:
            continue
        
        # d2 must be > d1 and 1/d2 < rem1
        # d2 > 1/rem1
        d2_min = d1 + 1
        # Upper bound for d2: we want reasonable denominators
        d2_max = min(200, int(rem1.denominator / rem1.numerator) + 50)
        d2_max = max(d2_max, d2_min)
        
        for d2 in range(d2_min, d2_max + 1):
            if time.time() - start_time > time_limit:
                break
            
            rem2 = rem1 - Fraction(1, d2)
            if rem2 < 0:
                continue
            
            if rem2 == 0:
                update_candidate([d1, d2])
                continue
            
            used = {d1, d2}
            completion = greedy_complete(rem2, used.copy(), 10000000)
            if completion is not None:
                all_denoms = [d1, d2] + completion
                if len(set(all_denoms)) == len(all_denoms) and len(all_denoms) <= 256:
                    update_candidate(all_denoms)
    
    # If we found a better solution, return it
    if best_max_den < 1736:
        return best_solution
    
    # Otherwise, return the known good solution
    return best_solution
# EVOLVE-BLOCK-END
