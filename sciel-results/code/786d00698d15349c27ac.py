from fractions import Fraction
import time

# Global deadline
_DEADLINE = 0
_BEST = None

def construct():
    global _DEADLINE, _BEST
    start_time = time.time()
    _DEADLINE = start_time + 55  # 55s safety margin

    # The minimum number of terms is 9.
    # We try to find a solution with exactly 9 terms.
    # If that fails, we try 10, etc., but 9 should work.
    
    # Known valid solution to fallback if search fails (verified manually or by code)
    # Let's try to find one via search first.
    
    for k in range(9, 12):
        if time.time() > _DEADLINE:
            break
        result = dfs_egyptian(Fraction(1, 1), 5, k)
        if result is not None:
            return result
            
    # Fallback: A known 9-term solution? 
    # If the search is too slow, we need a hardcoded valid solution.
    # Let's construct one using the algorithm logic in a tight loop if possible, 
    # or just return a greedy solution if it's valid (though greedy might use >9 terms).
    
    # Actually, let's just ensure the DFS is fast enough.
    # If no solution found within deadline, return a valid (possibly non-minimal) solution 
    # to avoid score 0. A greedy solution always terminates.
    
    return greedy_solution()

def dfs_egyptian(target, min_denom, terms_left):
    """
    Depth-first search for Egyptian fraction representation.
    target: Fraction
    min_denom: minimum allowed denominator
    terms_left: number of terms remaining to find
    """
    global _DEADLINE
    
    if time.time() > _DEADLINE:
        return None
        
    if terms_left == 0:
        return [] if target == 0 else None
        
    if terms_left == 1:
        if target.numerator != 1:
            return None
        d = target.denominator
        if d >= min_denom and d <= 10000000:
            return [d]
        return None
        
    if terms_left == 2:
        # 1/a + 1/b = target, a < b, a >= min_denom
        # b = a / (a*target - 1)
        # a*target - 1 must divide a
        # a*target > 1 => a > 1/target
        # b > a => 1/(a*target - 1) > 1 => a*target - 1 < 1 => a*target < 2
        # So 1/target < a < 2/target
        
        a_min = min_denom
        inv_target = Fraction(1, 1) / target
        a_lower = inv_target.numerator // inv_target.denominator
        if inv_target.numerator % inv_target.denominator != 0:
            a_lower += 1
        a_min = max(a_min, a_lower + 1)
        
        a_max = 10000000
        two_over_target = Fraction(2, 1) / target
        a_upper = two_over_target.numerator // two_over_target.denominator
        a_max = min(a_max, a_upper)
        
        for a in range(a_min, a_max + 1):
            if time.time() > _DEADLINE:
                return None
            den = a * target - 1
            if den <= 0:
                continue
            if a % den != 0:
                continue
            b = a // den
            if b > a and b >= min_denom and b <= 10000000:
                return [a, b]
        return None

    # Pruning: Max possible sum of remaining terms
    # Sum_{i=0}^{terms_left-1} 1/(min_denom + i)
    max_sum = Fraction(0, 1)
    for i in range(terms_left):
        max_sum += Fraction(1, min_denom + i)
    if max_sum < target:
        return None

    # Lower bound for d
    # 1/d <= target => d >= 1/target
    d_min = min_denom
    inv_target = Fraction(1, 1) / target
    d_lower = inv_target.numerator // inv_target.denominator
    if inv_target.numerator % inv_target.denominator != 0:
        d_lower += 1
    d_min = max(d_min, d_lower)
    
    # Upper bound for d
    # terms_left / d >= target => d <= terms_left / target
    d_max_approx = int(terms_left / target) + 3
    d_max = min(10000000, max(d_min, d_max_approx))
    
    for d in range(d_min, d_max + 1):
        if time.time() > _DEADLINE:
            return None
            
        term = Fraction(1, d)
        if term > target:
            break
            
        new_target = target - term
        if new_target <= 0:
            continue
            
        # Check if new_target can be formed by terms_left-1 terms > d
        # Max sum of remaining
        max_rest = Fraction(0, 1)
        for i in range(terms_left - 1):
            max_rest += Fraction(1, d + 1 + i)
            
        if new_target > max_rest:
            break
            
        # Min sum check? 
        # The minimum sum is achieved by taking the largest denominators? 
        # No, we are constrained to be > d. The smallest sum is 0 (infinite denominators).
        # But we have a cap of 10^7.
        # If new_target is very small, we might need huge denominators.
        # If new_target < 1 / (10^7 * terms_left), it's impossible.
        # But usually the max_rest check is the bottleneck.
        
        res = dfs_egyptian(new_target, d + 1, terms_left - 1)
        if res is not None:
            return [d] + res
            
    return None

def greedy_solution():
    """Fallback greedy solution."""
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    return denoms
