"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with safety margin
    
    # We want to find the minimum number of distinct denominators >= 5
    # whose reciprocals sum to exactly 1.
    
    # Try k = 3, 4, 5, 6, 7, ... until we find a solution or hit deadline
    for k in range(3, 10):
        if time.time() > deadline:
            break
        
        # Backtracking search for k denominators
        result = _search_k_terms(k, start_time, deadline)
        if result is not None:
            return result
    
    # Fallback: return a known valid solution if we didn't find optimal
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 is not valid, let me use a known one
    # Actually, let's just return any valid solution we can construct
    # 1 = 1/5 + 1/5 + ... no, must be distinct
    
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35? Let me compute properly
    # Let me just use a constructive approach for fallback
    # 1/2 = 1/3 + 1/6, so 1 = 1/2 + 1/2 but can't use same
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + ... 
    
    # Safe fallback: use the DP approach from original as backup
    return _fallback_dp()

def _search_k_terms(k, start_time, deadline):
    """Search for k distinct denominators >= 5 summing reciprocals to 1."""
    
    def backtrack(remaining_terms, min_denom, remaining_sum, current):
        # remaining_sum is a Fraction
        if time.time() > deadline:
            return None
        
        if remaining_terms == 0:
            if remaining_sum == 0:
                return list(current)
            return None
        
        if remaining_terms == 1:
            # Need 1/d = remaining_sum, so d = 1/remaining_sum
            if remaining_sum > 0:
                d = 1 / remaining_sum
                if d.denominator == 1 and d.numerator >= min_denom and d.numerator <= 10000000:
                    current.append(d.numerator)
                    return list(current)
            return None
        
        # Pruning: max possible sum with remaining_terms terms starting from min_denom
        # is 1/min_denom + 1/(min_denom+1) + ... + 1/(min_denom + remaining_terms - 1)
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible += Fraction(1, min_denom + i)
        if max_possible < remaining_sum:
            return None
        
        # Min possible sum: use largest denominators... but that's unbounded.
        # Instead: the smallest denominator we can pick is min_denom.
        # If 1/min_denom > remaining_sum, we need a larger denominator.
        # Also, we need remaining_sum to be representable.
        
        # Lower bound on next denominator:
        # 1/d <= remaining_sum, so d >= 1/remaining_sum
        # Also d >= min_denom
        # And we need to be able to fill remaining_terms - 1 more terms
        # with denominators > d, whose max sum is 1/(d+1) + ... + 1/(d + remaining_terms - 1)
        
        low_d = max(min_denom, (1 / remaining_sum).numerator if remaining_sum > 0 else min_denom)
        # Actually 1/remaining_sum might not be integer
        if remaining_sum > 0:
            low_d = max(min_denom, -(-1 // remaining_sum.numerator * remaining_sum.denominator))  # ceil(1/remaining_sum)
            # Simpler: low_d = max(min_denom, int(1 / remaining_sum) + 1) if not exact
            # Let me just use: d must satisfy 1/d <= remaining_sum => d >= 1/remaining_sum
            # So d >= ceil(1/remaining_sum) if 1/remaining_sum is not integer, else 1/remaining_sum
            inv = Fraction(1, 1) / remaining_sum
            if inv.denominator == 1:
                low_d = max(min_denom, inv.numerator)
            else:
                low_d = max(min_denom, inv.numerator // inv.denominator + 1)
        
        # Upper bound on next denominator:
        # We need remaining_terms - 1 more terms with denominators > d.
        # The minimum sum they can contribute is... well, they can be arbitrarily large,
        # so the sum can be arbitrarily close to 0. But we need exactly remaining_sum - 1/d.
        # The constraint is that remaining_sum - 1/d > 0, so 1/d < remaining_sum, d > 1/remaining_sum.
        # Also, the remaining_terms - 1 terms must sum to remaining_sum - 1/d.
        # Each of those is at most 1/(d+1), 1/(d+2), ..., so max is sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # We need remaining_sum - 1/d <= that max.
        
        # For efficiency, let's just iterate d from low_d upward with a reasonable bound
        # Upper bound: d can't be too large because we need remaining_sum - 1/d to be
        # achievable. A rough upper bound: d < 1/remaining_sum * remaining_terms (rough)
        
        # Let's set an upper bound to keep search tractable
        upper_d = min(10000000, low_d + 100000)  # Cap the search range per level
        
        for d in range(low_d, upper_d + 1):
            if time.time() > deadline:
                return None
            
            frac_d = Fraction(1, d)
            if frac_d >= remaining_sum:
                continue
            
            new_remaining = remaining_sum - frac_d
            if new_remaining < 0:
                continue
            
            # Check if new_remaining is achievable with remaining_terms - 1 terms > d
            # Max possible: sum_{i=1}^{remaining_terms-1} 1/(d+i)
            max_rem = Fraction(0, 1)
            for i in range(1, remaining_terms):
                max_rem += Fraction(1, d + i)
            if new_remaining > max_rem:
                continue
            
            current.append(d)
            result = backtrack(remaining_terms - 1, d + 1, new_remaining, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    return backtrack(k, 5, Fraction(1, 1), [])

def _fallback_dp():
    """Original DP approach as fallback."""
    target = 2520
    reached = {0: []}
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
# EVOLVE-BLOCK-END
