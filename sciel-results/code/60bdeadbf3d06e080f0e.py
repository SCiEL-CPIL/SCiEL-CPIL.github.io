"""Solve: factor 12! into 1..12 DISTINCT integers > 12, minimize the LARGEST factor."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def _all_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i > 12:
                divs.append(i)
            if i != n // i and n // i > 12:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    deadline = time.time() + 85  # safety margin under 90s kill
    target = factorial(12)
    
    # Get all divisors of target that are > 12
    divs = _all_divisors(target)
    
    # Binary search / iterative approach: try increasing max factor M
    # For each M, check if we can find a valid decomposition with all factors <= M
    # Start from a reasonable lower bound and increase
    
    best_result = [target]  # fallback: single factor
    
    # The minimum possible max factor is at least ceil(target^(1/12)) since we can have at most 12 factors
    # But we also need factors > 12, so let's start from 13
    # Actually, let's be smarter: use the fact that we need product = 12!
    # With k factors all > 12, the minimum possible max is bounded below by (12!)^(1/k)
    
    # Let's try M values in increasing order
    # We need to find the smallest M such that a valid decomposition exists with all factors <= M
    
    # Generate candidate M values - divisors of target that are > 12
    # Sort them and try each as the maximum
    candidate_maxes = sorted([d for d in divs if d > 12])
    
    for M in candidate_maxes:
        if time.time() > deadline:
            break
        
        # Check if we can decompose target into distinct factors, all in (12, M]
        # with at most 12 factors
        result = _search_decomposition(target, divs, M, deadline)
        if result is not None:
            best_result = result
            break  # Found the smallest M, this is optimal
    
    return best_result

def _search_decomposition(target, divs, max_factor, deadline):
    """
    Search for a decomposition of target into distinct factors, all > 12 and <= max_factor.
    Returns a list of factors or None if not found.
    """
    # Filter divisors to those <= max_factor
    valid_divs = [d for d in divs if d <= max_factor]
    
    if not valid_divs:
        return None
    
    # DFS with pruning
    def dfs(remaining, start_idx, current_factors):
        if time.time() > deadline:
            return None
        
        if remaining == 1:
            if len(current_factors) >= 1 and len(current_factors) <= 12:
                return list(current_factors)
            return None
        
        if len(current_factors) >= 12:
            return None
        
        # Pruning: if remaining > 1, we need at least one more factor
        # The smallest available factor is valid_divs[start_idx]
        # If remaining is not divisible by any valid divisor from start_idx onward, prune
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            if remaining % d == 0:
                new_remaining = remaining // d
                # Pruning: new_remaining must be 1 or have factors all <= max_factor
                # Also, if new_remaining > 1, it must be possible to factor it
                # Quick check: new_remaining must be >= 13 (since all factors > 12)
                # unless new_remaining == 1
                
                if new_remaining == 1:
                    current_factors.append(d)
                    if len(current_factors) >= 1 and len(current_factors) <= 12:
                        result = list(current_factors)
                        current_factors.pop()
                        return result
                    current_factors.pop()
                    continue
                
                # If new_remaining > 1, we need more factors
                # The product of remaining factors must equal new_remaining
                # All those factors must be > 12 and <= max_factor
                # So new_remaining must be factorable into distinct factors in (12, max_factor]
                
                # Pruning: if new_remaining < 13, impossible (need at least one factor > 12)
                if new_remaining < 13:
                    continue
                
                # Pruning: new_remaining must be divisible by some combination of valid_divs[i+1:]
                # Quick necessary condition: new_remaining must have a divisor > 12
                has_valid_divisor = False
                for j in range(i + 1, len(valid_divs)):
                    if new_remaining % valid_divs[j] == 0:
                        has_valid_divisor = True
                        break
                if not has_valid_divisor:
                    continue
                
                current_factors.append(d)
                result = dfs(new_remaining, i + 1, current_factors)
                current_factors.pop()
                if result is not None:
                    return result
        
        return None
    
    return dfs(target, 0, [])
# EVOLVE-BLOCK-END
