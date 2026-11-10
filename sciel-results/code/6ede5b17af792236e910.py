import time
from fractions import Fraction

def _find_solution_11(min_den=5, max_den=10000000, deadline=None):
    """
    Find 11 distinct denominators >= min_den whose reciprocal sum is exactly 1.
    Strategy: pick 9 denominators, solve for last 2.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    target = Fraction(1, 1)
    
    def try_solve(dens):
        """Try to solve for last 2 given 9 denominators."""
        current_sum = sum(Fraction(1, d) for d in dens)
        if current_sum >= target:
            return None
        R = target - current_sum
        
        min_a = dens[-1] + 1
        # R is a Fraction. 1/R is a Fraction.
        # We need 1/a + 1/b = R with a < b.
        # 1/b = R - 1/a = (R*a - 1)/a
        # b = a / (R*a - 1)
        # For b to be integer, R*a - 1 must divide a.
        # Also b > a implies a / (R*a - 1) > a => 1 / (R*a - 1) > 1 => R*a - 1 < 1 => R*a < 2 => a < 2/R.
        # And R*a - 1 > 0 => a > 1/R.
        
        low_a_frac = Fraction(1, 1) / R
        high_a_frac = Fraction(2, 1) / R
        
        low_a = max(min_a, low_a_frac.numerator // low_a_frac.denominator + 1)
        high_a = min(high_a_frac.numerator // high_a_frac.denominator, max_den)
        
        if high_a < low_a:
            return None
        
        dens_set = set(dens)
        for a in range(low_a, high_a + 1):
            if a in dens_set:
                continue
            Ra_minus_1 = R * a - 1
            if Ra_minus_1 <= 0:
                continue
            b_frac = Fraction(a, 1) / Ra_minus_1
            if b_frac.denominator == 1:
                b = b_frac.numerator
                if b > a and b <= max_den and b not in dens_set:
                    return dens + [a, b]
        return None
    
    def search(dens, idx, partial_sum):
        if time.time() > deadline:
            return None
        
        if idx == 9:
            return try_solve(dens)
        
        start = dens[idx - 1] + 1 if idx > 0 else min_den
        
        # Pruning:
        # 1. If partial_sum >= target, break (handled by caller or here)
        # 2. If partial_sum + max_possible_remaining < target, prune.
        #    Max possible remaining sum is sum of reciprocals of next 'remaining' integers.
        # 3. If partial_sum + min_possible_remaining > target, prune.
        #    Min possible remaining sum is sum of reciprocals of largest possible integers?
        #    No, we are choosing increasing integers. The smallest sum for 'remaining' terms 
        #    starting from 'start' is actually not bounded below by 0 in a useful way for pruning 
        #    unless we know the max denominator. But we know max_den.
        #    However, the "min sum" for distinct increasing integers starting from 'start' 
        #    is achieved by taking the largest possible denominators? No, that's not right.
        #    The smallest sum is achieved by taking the largest possible denominators, which is 
        #    max_den - remaining + 1 ... max_den.
        #    But we are constrained to be > last chosen.
        #    Actually, if we just want to check if it's *possible* to reach target,
        #    the lower bound on the sum of the remaining terms is 0 (since we can pick huge denominators).
        #    The upper bound is sum(1/i for i in range(start, start+remaining)).
        
        remaining = 9 - idx
        if remaining > 0:
            # Max sum of remaining terms
            max_add = sum(Fraction(1, i) for i in range(start, start + remaining))
            if partial_sum + max_add < target:
                return None
            
            # Min sum of remaining terms
            # We can pick denominators up to max_den.
            # The smallest sum is obtained by picking the largest possible denominators.
            # But we must be > dens[-1] and distinct.
            # The largest possible denominators are max_den, max_den-1, ..., max_den-remaining+1.
            # But they must be > start-1 (which is dens[-1] or min_den-1).
            # If max_den - remaining + 1 <= start - 1, then we are constrained.
            # But max_den is 10^7, so usually we can pick large denominators.
            # min_add = sum(Fraction(1, max_den - i) for i in range(remaining))
            # if partial_sum + min_add > target: return None
            # This is rarely true because min_add is tiny.
        
        # Heuristic bounds for the next denominator
        # We want to keep denominators small to make the sum close to 1.
        # But we also need enough room for the last 2 terms.
        
        # Let's estimate the required range for d.
        # If we pick d, the remaining sum needed is target - (partial_sum + 1/d).
        # This must be achievable by (remaining-1) terms.
        
        # Simple heuristic: limit the range of d to keep search space small.
        # For idx 0, d is around 5-10.
        # For idx 8, d is larger.
        
        # Let's use a dynamic range.
        if idx < 3:
            end = start + 30
        elif idx < 6:
            end = start + 50
        else:
            end = start + 100
            
        end = min(end, max_den)
        
        for d in range(start, end + 1):
            if time.time() > deadline:
                return None
            
            new_partial = partial_sum + Fraction(1, d)
            if new_partial >= target:
                break
            
            # Additional pruning: check if new_partial + max_possible_remaining_2 < target
            rem_2 = 9 - (idx + 1)
            if rem_2 > 0:
                max_add_2 = sum(Fraction(1, i) for i in range(d + 1, d + 1 + rem_2))
                if new_partial + max_add_2 < target:
                    continue
            
            result = search(dens + [d], idx + 1, new_partial)
            if result is not None:
                return result
        
        return None
    
    return search([], 0, Fraction(0, 1))

def construct():
    deadline = time.time() + 55
    
    res = _find_solution_11(deadline=deadline)
    if res is not None:
        return res
    
    # Fallback to seed logic (2520 divisors)
    # This is guaranteed to find a solution, but might have many terms.
    # The task asks to minimize count.
    # If we fail to find 11, we return the fallback.
    # The fallback in the seed program finds a subset of divisors of 2520 summing to 2520 (which means sum of 1/d = 1).
    # This is a valid solution, but likely has more than 11 terms.
    
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
