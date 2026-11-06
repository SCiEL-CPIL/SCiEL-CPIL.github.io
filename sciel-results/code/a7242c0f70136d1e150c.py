"""Solve: factor 12! into 1..12 DISTINCT integers > 12, minimize the LARGEST factor."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from collections import Counter

def construct():
    deadline = time.time() + 85  # safety margin under 90s kill
    target = factorial(12)
    
    # Prime factorization of 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We need to find distinct divisors d_i > 12 with product = target
    # Minimize max(d_i)
    
    # Strategy: For each candidate maximum M (in increasing order), check if we can
    # find a valid decomposition with all factors in (12, M].
    # Use a smarter search: precompute all divisors > 12, then use backtracking with
    # strong pruning based on prime exponents.
    
    # All divisors of 12! that are > 12
    divs = []
    i = 13
    while i * i <= target:
        if target % i == 0:
            divs.append(i)
            other = target // i
            if other > 12 and other != i:
                divs.append(other)
        i += 1
    if target > 12:
        divs.append(target)
    divs = sorted(set(divs))
    
    # For each divisor, compute its prime exponent vector
    def prime_exponents(n):
        """Return dict of prime -> exponent for n"""
        result = {}
        for p in [2, 3, 5, 7, 11]:
            while n % p == 0:
                result[p] = result.get(p, 0) + 1
                n //= p
        return result
    
    # Precompute exponent vectors for all divisors
    div_exp = {}
    for d in divs:
        div_exp[d] = prime_exponents(d)
    
    # Target exponent vector
    target_exp = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}
    
    # Try increasing M values
    # The minimum possible M is at least 13
    # We iterate M over the sorted divisors
    for M in divs:
        if time.time() > deadline:
            break
        
        # Filter divisors to those <= M
        valid_divs = [d for d in divs if d <= M]
        
        if len(valid_divs) < 1:
            continue
        
        # Check if we can decompose target into distinct factors from valid_divs
        result = _search_decomposition(target, target_exp, valid_divs, div_exp, deadline)
        if result is not None:
            return result
    
    # Fallback: single factor
    return [target]

def _search_decomposition(target, target_exp, valid_divs, div_exp, deadline):
    """
    Search for a decomposition of target into distinct factors from valid_divs.
    All factors must be > 12 (guaranteed by construction).
    Returns a list of factors or None if not found.
    """
    n = len(valid_divs)
    
    # Backtracking with prime exponent tracking
    # We track remaining exponents
    remaining_exp = dict(target_exp)
    
    def dfs(start_idx, current_factors):
        if time.time() > deadline:
            return None
        
        # Check if we've used up all exponents
        if all(remaining_exp.get(p, 0) == 0 for p in [2, 3, 5, 7, 11]):
            if len(current_factors) >= 1 and len(current_factors) <= 12:
                return list(current_factors)
            return None
        
        if len(current_factors) >= 12:
            return None
        
        # Pruning: check if remaining exponents can be satisfied by remaining divisors
        # For each prime, sum the exponents in remaining divisors
        # This is expensive, so do a quick check
        for i in range(start_idx, n):
            d = valid_divs[i]
            exp_d = div_exp[d]
            
            # Check if this divisor's exponents don't exceed remaining
            can_use = True
            for p in [2, 3, 5, 7, 11]:
                if exp_d.get(p, 0) > remaining_exp.get(p, 0):
                    can_use = False
                    break
            
            if not can_use:
                continue
            
            # Try using this divisor
            for p in [2, 3, 5, 7, 11]:
                remaining_exp[p] -= exp_d.get(p, 0)
            
            current_factors.append(d)
            result = dfs(i + 1, current_factors)
            current_factors.pop()
            
            for p in [2, 3, 5, 7, 11]:
                remaining_exp[p] += exp_d.get(p, 0)
            
            if result is not None:
                return result
        
        return None
    
    return dfs(0, [])
# EVOLVE-BLOCK-END
