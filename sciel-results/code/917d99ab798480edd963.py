"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd
from functools import lru_cache
from collections import defaultdict

TARGET = factorial(12)  # 479001600

def _all_divisors(n):
    """Generate all divisors of n."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def _valid_divisors(max_val):
    """Get divisors of TARGET that are > 12 and <= max_val."""
    return [d for d in _all_divisors(TARGET) if 12 < d <= max_val]

def _can_complete(remaining, max_factor, min_factor, max_factors_left):
    """
    Check if 'remaining' can be factored into at most max_factors_left distinct factors,
    each between min_factor and max_factor inclusive.
    This is a necessary condition (not sufficient) for pruning.
    """
    if remaining == 1:
        return True
    if max_factors_left <= 0:
        return False
    if remaining > max_factor ** max_factors_left:
        return False
    if remaining < min_factor:
        return False
    # If remaining is prime, it must be used as a single factor
    # Check if remaining is prime
    if remaining > 1:
        is_prime = True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if p * p > remaining:
                break
            if remaining % p == 0:
                is_prime = False
                break
        if is_prime:
            return min_factor <= remaining <= max_factor
    return True

def _dfs(remaining, min_factor, max_factor, factors_left, used, path, best, deadline):
    """
    DFS to find a valid factorization.
    remaining: product still to be achieved
    min_factor: minimum allowed factor value
    max_factor: maximum allowed factor value (current best largest)
    factors_left: maximum number of factors still allowed
    used: set of already used factors
    path: current list of factors
    best: list to store the best solution found
    deadline: time limit
    """
    if time.time() > deadline:
        return
    
    if remaining == 1:
        if len(path) > 0 and len(path) <= 12:
            # Found a valid solution
            largest = max(path)
            if largest < max(best[0], default=float('inf')):
                best[0] = largest
                best[1] = list(path)
        return
    
    if factors_left <= 0:
        return
    
    # Try each possible factor from min_factor to max_factor
    # But we need it to divide remaining
    # Get divisors of remaining that are in [min_factor, max_factor]
    # and not in used
    
    # For efficiency, iterate through possible divisors
    # We can generate divisors of remaining on the fly
    # But since remaining can be large, let's use a smarter approach
    
    # Actually, let's precompute divisors of remaining up to max_factor
    # But that's expensive. Instead, iterate through candidates.
    
    # Better approach: iterate through divisors of remaining
    # that are >= min_factor and <= max_factor
    
    # Generate divisors of remaining
    rem_divs = _all_divisors(remaining)
    
    for d in rem_divs:
        if d < min_factor or d > max_factor:
            continue
        if d in used:
            continue
        if time.time() > deadline:
            return
        
        # Try using d as a factor
        new_remaining = remaining // d
        new_min = d + 1  # factors must be distinct and we're going in increasing order
        # Actually, we should allow any order, but to avoid duplicates,
        # we can enforce increasing order
        
        # Check if new_remaining can be completed
        if not _can_complete(new_remaining, max_factor, new_min, factors_left - 1):
            continue
        
        used.add(d)
        path.append(d)
        _dfs(new_remaining, new_min, max_factor, factors_left - 1, used, path, best, deadline)
        path.pop()
        used.remove(d)
        
        # Update best if we found a better solution
        # (handled in the base case)
        
        # Pruning: if we've already found a solution with largest factor <= d,
        # we can stop trying larger d values
        if best[0] is not None and d >= best[0]:
            break

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # First, try to find a solution with a small maximum factor
    # Binary search on the maximum factor
    
    # Lower bound: at least 13 (smallest factor > 12)
    # Upper bound: TARGET itself (single factor)
    
    # Let's try to find the minimum possible largest factor
    # We'll do a binary search on the max factor value
    
    # But binary search requires a decision procedure:
    # "Can we factor TARGET into distinct factors all in [13, M]?"
    
    # Let's use DFS with pruning
    
    # Start with a high max_factor and try to reduce it
    # Or use binary search
    
    # For now, let's do a greedy/DFS approach with a reasonable max_factor
    
    # The product is 12! = 479001600
    # If we use k factors, each at most M, then M^k >= 479001600
    # For k=3: M >= 783
    # For k=4: M >= 262
    # For k=5: M >= 137
    # For k=6: M >= 90
    # For k=7: M >= 66
    # For k=8: M >= 51
    # For k=9: M >= 42
    # For k=10: M >= 35
    # For k=11: M >= 30
    # For k=12: M >= 27
    
    # So we can try M from 27 upward
    
    # Let's try increasing M until we find a valid factorization
    
    best_solution = [float('inf'), []]
    
    # Try different max_factor values
    for max_f in range(27, 500):
        if time.time() > deadline:
            break
        
        # Try to find a factorization with all factors <= max_f
        # Use DFS with min_factor = 13
        used = set()
        path = []
        _dfs(TARGET, 13, max_f, 12, used, path, best_solution, deadline)
        
        # If we found a solution with largest factor <= max_f, we can stop
        # because we're trying increasing max_f
        if best_solution[0] <= max_f:
            break
    
    # If no solution found, return the best we have or a fallback
    if best_solution[1]:
        return best_solution[1]
    
    # Fallback: return a single factor (which is valid but not optimal)
    return [TARGET]
# EVOLVE-BLOCK-END
