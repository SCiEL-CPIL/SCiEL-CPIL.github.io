import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    denominators = list(range(2, 49))
    
    # dp maps sum -> (prev_sum, index)
    # Initialize with sum 0
    dp = {Fraction(0): (None, -1)}
    
    for i, d in enumerate(denominators):
        val = Fraction(1, d)
        # Snapshot current keys to avoid using the current item multiple times
        # and to ensure we only extend sums that were achievable before adding item i
        current_keys = list(dp.keys())
        
        for s in current_keys:
            new_s = s + val
            if new_s < 1:
                if new_s not in dp:
                    dp[new_s] = (s, i)
                    
        if time.time() > deadline:
             # Fallback to greedy if DP is unexpectedly slow
             selected, total = [], Fraction()
             for j, dd in enumerate(denominators):
                 if total + Fraction(1, dd) < 1:
                     selected.append(j)
                     total += Fraction(1, dd)
             return selected

    # Find the sum closest to 1 (strictly below)
    # Since all sums in dp are < 1, max is the best
    best_sum = max(dp.keys())
    
    # Reconstruct indices
    indices = []
    curr = best_sum
    while curr is not None:
        prev_sum, idx = dp[curr]
        if idx != -1:
            indices.append(idx)
        curr = prev_sum
        
    return indices
