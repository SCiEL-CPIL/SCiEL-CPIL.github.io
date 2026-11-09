import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _validate(denoms):
    """Validate a list of denominators. Return max if valid, else None."""
    if not denoms or len(denoms) > MAX_TERMS:
        return None
    if len(set(denoms)) != len(denoms):
        return None
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return None
    total = sum(Fraction(1, d) for d in denoms)
    if total != TARGET:
        return None
    return max(denoms)

def _search_for_M(M, deadline):
    """
    Check if 5/31 - 1/M can be represented as sum of distinct unit fractions
    with denominators in [1, M-1].
    Returns list of denominators or None.
    """
    rem = TARGET - Fraction(1, M)
    if rem <= 0:
        return None
    
    # We need to represent 'rem' using denominators < M.
    # DFS state: (current_remainder, min_denom, max_denom, used_set)
    # To speed up, we can use integer arithmetic if we scale by LCM, but Fractions are easier.
    
    def dfs(rem, min_d, max_d, path):
        if time.time() > deadline:
            return None
        if rem == 0:
            return list(path)
        if not path:
            # Heuristic: estimate number of terms needed
            # Max sum with k terms starting from 1 is H_k.
            # If rem is very small, we need many terms? No, small denominators give large sums.
            # To get small sum, we need large denominators.
            # Min sum with k terms using largest available (max_d, max_d-1, ...)
            # If rem < min_possible_sum(k), we need more terms? No, we can't have more than max_d-min_d+1 terms.
            pass
            
        # Lower bound for next denominator:
        # 1/d <= rem  => d >= 1/rem
        min_d_eff = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound for next denominator:
        # We must be able to form the remainder with the remaining terms.
        # If we pick d, new rem is rem - 1/d.
        # The next terms must be > d and <= max_d.
        # If rem - 1/d > 0, we need at least one more term.
        # The largest possible next term is 1/(d+1).
        # So we need rem - 1/d <= sum of all possible future terms?
        # Simpler: If we pick d, and rem - 1/d > 0, we need to represent it with denominators in [d+1, max_d].
        # The minimum value of the next term is 1/max_d.
        # So if rem - 1/d < 1/max_d, it's impossible (unless rem-1/d == 0, handled above).
        # Also, if rem - 1/d > 0, we need at least one term.
        # The maximum value of the next term is 1/(d+1).
        # So we must have rem - 1/d <= 1/(d+1) + 1/(d+2) + ...
        # A tighter bound: If we have to pick exactly one more term, it must be 1/(rem-1/d) which must be integer and in range.
        # If we can pick more, the sum of the largest available denominators must be >= rem - 1/d.
        # Sum of largest k denominators approx log(max_d) - log(max_d-k).
        
        # Let's just iterate d.
        # To prune: if rem > sum(1/i for i in range(min_d_eff, max_d+1)), return None.
        # This sum is expensive to compute every time.
        # Approx: H_max_d - H_min_d_eff.
        
        # Pruning 1: If rem is too large to be formed by all available denominators from min_d_eff to max_d
        # Max sum = H(max_d) - H(min_d_eff - 1)
        # We can precompute harmonic numbers? Or just estimate.
        # For M < 248, the range is small.
        
        # Pruning 2: If rem is too small to be formed by the largest available denominators?
        # No, we can always pick smaller denominators to increase the sum.
        # Wait, we are constrained to be DISTINCT and > min_d.
        # The smallest sum we can form with ANY number of terms from [min_d, max_d] is 1/max_d (picking just max_d).
        # If rem < 1/max_d, we can't form it (since all terms are <= 1/max_d? No, 1/min_d is larger).
        # The smallest term is 1/max_d. The largest term is 1/min_d.
        # If rem < 1/max_d, impossible.
        if rem < Fraction(1, max_d):
            return None
            
        for d in range(min_d_eff, max_d + 1):
            if rem - Fraction(1, d) < 0:
                break # Since d is increasing, 1/d is decreasing, rem-1/d is increasing.
                # Wait. If rem - 1/d < 0, then 1/d > rem.
                # For larger d, 1/d is smaller, so rem - 1/d becomes less negative, then positive.
                # So we should CONTINUE, not break.
                # Actually, if 1/d > rem, then rem - 1/d < 0.
                # We need rem - 1/d >= 0.
                # So we need 1/d <= rem => d >= 1/rem.
                # This is exactly min_d_eff.
                # So the loop starts at min_d_eff, so rem - 1/d >= 0 always.
            
            rem_new = rem - Fraction(1, d)
            if rem_new == 0:
                return path + [d]
            
            # Recurse
            # Next denominators must be > d and <= max_d
            res = dfs(rem_new, d + 1, max_d, path + [d])
            if res is not None:
                return res
        return None

    return dfs(rem, 1, M - 1, [])

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Try to find solutions with max < best_max
    # Iterate M downwards from best_max - 1
    for M in range(best_max - 1, 1, -1):
        if time.time() > deadline:
            break
        
        res = _search_for_M(M, deadline)
        if res is not None:
            full_sol = res + [M]
            # Validate to be sure
            mx = _validate(full_sol)
            if mx is not None:
                best_max = mx
                best_solution = full_sol
                # Since we iterate M downwards, the first valid M is the minimum possible max denominator?
                # Yes, because we are checking if a solution EXISTS with max exactly M.
                # If it exists, M is a candidate for the minimum max denominator.
                # Since we go down from 247, the first one we find is the smallest M < 248.
                return best_solution
                
    return best_solution
