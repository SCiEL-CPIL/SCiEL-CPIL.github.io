"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def _dfs(remaining, min_d, k, deadline, chosen):
    """
    Recursively find k distinct denominators >= min_d whose reciprocals sum to 'remaining'.
    Returns a list of denominators or None if no solution found within deadline.
    """
    if time.time() > deadline:
        return None
    
    if k == 0:
        if remaining == 0:
            return chosen
        return None
    
    if remaining <= 0:
        return None
        
    # Pruning: Max possible sum with k terms starting from min_d
    # Sum_{i=0}^{k-1} 1/(min_d + i)
    max_possible = Fraction(0)
    for i in range(k):
        max_possible += Fraction(1, min_d + i)
    if max_possible < remaining:
        return None
        
    # Pruning: Min possible sum with k terms starting from min_d
    # The denominators must be distinct and >= min_d.
    # To minimize the sum, we pick the largest possible denominators? No, we are constrained by the search space.
    # Actually, a better lower bound check:
    # The smallest possible sum for k distinct denominators >= min_d is not simply k/max_denom because 
    # we don't have a fixed max_denom for the *search* other than 10^7, but the denominators must be 
    # large enough to not overshoot.
    # However, a simpler necessary condition:
    # 1/d_1 + ... + 1/d_k = remaining
    # Since d_i are distinct integers >= min_d, the smallest sum is achieved by the largest denominators, 
    # but we don't have an upper bound on denominators in the search space other than 10^7.
    # But we can use: remaining <= k / min_d (already checked via max_possible? No, max_possible is sum of smallest denominators).
    # Wait, max_possible used min_d, min_d+1... which gives the LARGEST sum.
    # We need to ensure we don't exceed 'remaining' with the SMALLEST denominators? No, we subtract.
    # We need to ensure that even with the LARGEST possible denominators (up to 10^7), we can reach 'remaining'.
    # Min sum = sum of 1/(10^7 - i) for i=0..k-1? That's very small.
    # Actually, the constraint is:
    # 1/d_i <= remaining / k  => d_i >= k / remaining
    # So min_d_effective = max(min_d, ceil(k / remaining))
    
    # Let's refine the range for the next denominator d
    # 1/d <= remaining  => d >= 1/remaining
    # 1/d >= remaining / k => d <= k / remaining
    # Also d >= min_d
    
    d_min = max(min_d, math.ceil(remaining.denominator / remaining.numerator))
    d_max = int(remaining.numerator * k / remaining.denominator)
    
    if d_min > d_max:
        return None
        
    # Iterate d from d_min to d_max
    # Optimization: If d_max is huge, we might need to be careful.
    # But d_max is bounded by k/remaining.
    # For k=7 and remaining ~ 0.3, d_max ~ 23.
    # For k=7 and remaining ~ 0.01, d_max ~ 700.
    # This range is manageable.
    
    for d in range(d_min, d_max + 1):
        if time.time() > deadline:
            return None
        next_remaining = remaining - Fraction(1, d)
        if next_remaining < 0:
            continue
        # If next_remaining is 0, we need 0 more terms, but we have k-1 left.
        # So if k-1 > 0, this is invalid unless next_remaining == 0 and k-1 == 0.
        # The base case handles k==0.
        
        # Pruning for the recursive call:
        # Can we make next_remaining with k-1 terms >= d+1?
        # Max sum for k-1 terms starting from d+1:
        max_possible_next = Fraction(0)
        for i in range(k - 1):
            max_possible_next += Fraction(1, d + 1 + i)
        if max_possible_next < next_remaining:
            continue
            
        # Also, min sum check is less critical here as we are constrained by d_min/d_max logic,
        # but let's add a simple check: next_remaining must be positive (handled) 
        # and we must be able to reach it.
        
        res = _dfs(next_remaining, d + 1, k - 1, deadline, chosen + [d])
        if res is not None:
            return res
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s safety margin for 60s limit
    
    # As analyzed, k < 7 is impossible for distinct denominators >= 5.
    # Start searching from k=7.
    # If k=7 fails (unlikely, but possible due to integer constraints), try k=8, etc.
    
    target = Fraction(1, 1)
    
    for k in range(7, 15):
        if time.time() > deadline:
            break
            
        # Initial min_denom is 5
        # We want to find k distinct denominators >= 5
        result = _dfs(target, 5, k, deadline, [])
        
        if result is not None:
            # Verify result
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result
                
    # Fallback: If DFS fails (should not happen for k>=7 with sufficient time),
    # try a known solution or a simpler DP if time permits.
    # However, k=7 should be solvable.
    # Example solution for k=7:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/420 ?
    # Sum = 1/5+1/6+1/7+1/8+1/9+1/10 = 0.955...
    # Remainder = 1 - 0.955... = 0.044...
    # 1/420 is approx 0.0023. Too small.
    # Let's rely on the DFS.
    
    # If we reach here, we failed to find a solution within time.
    # Return a valid, though possibly non-minimal, solution if one is hardcoded?
    # The task requires minimizing count. If we can't find min, we should return best effort.
    # But k=7 is the theoretical minimum.
    
    raise RuntimeError("No solution found within time limit")

# EVOLVE-BLOCK-END
