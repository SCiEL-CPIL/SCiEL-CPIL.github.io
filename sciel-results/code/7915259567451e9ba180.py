import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators to use integer arithmetic
    # LCM(2..48)
    L = 1
    for d in denominators:
        L = math.lcm(L, d)
    
    # Weights for each denominator
    weights = [L // d for d in denominators]
    
    # DP: dict mapping sum (int) to mask (int)
    # We want to maximize sum < L
    dp = {0: 0}
    
    # Check time and state explosion
    max_states = 2000000  # Arbitrary limit to prevent OOM/timeout
    
    for i in range(n):
        w = weights[i]
        new_dp = dict(dp)
        
        # Check if we are running out of time or states
        if time.time() > deadline - 10.0 or len(dp) > max_states:
            # Fallback to heuristic if DP is too slow
            return _heuristic_fallback(denominators)
            
        for s, m in dp.items():
            new_s = s + w
            if new_s < L:
                new_m = m | (1 << i)
                # Only store if this sum wasn't already reached
                # If it was, the existing mask is fine
                if new_s not in new_dp:
                    new_dp[new_s] = new_m
        dp = new_dp
        
        if time.time() > deadline - 5.0:
            # Run out of time in the middle of DP
            # Use the best found in current dp
            best_s = max(dp.keys())
            mask = dp[best_s]
            indices = [i for i in range(n) if (mask >> i) & 1]
            return indices

    # Find the best sum
    if not dp:
        return [0] # Should not happen
        
    best_s = max(dp.keys())
    best_mask = dp[best_s]
    
    indices = [i for i in range(n) if (best_mask >> i) & 1]
    return indices

def _heuristic_fallback(denominators):
    # Greedy + Local Search
    n = len(denominators)
    selected = set()
    current_sum = 0.0
    
    # Greedy: add items that keep sum < 1, prioritizing larger reciprocals?
    # Actually, to get close to 1, we should try to fill the gap.
    # Simple greedy: sort by reciprocal descending, add if fits.
    items = sorted(range(n), key=lambda i: -1.0/denominators[i])
    
    for i in items:
        if current_sum + 1.0/denominators[i] < 1.0:
            selected.add(i)
            current_sum += 1.0/denominators[i]
            
    # Local Search: try to improve
    improved = True
    while improved and time.time() < time.time() + 40: # Rough check
        improved = False
        # Try removing one item
        for i in list(selected):
            new_sum = current_sum - 1.0/denominators[i]
            # After removal, try adding one unselected item
            for j in range(n):
                if j not in selected:
                    if new_sum + 1.0/denominators[j] < 1.0:
                        new_total = new_sum + 1.0/denominators[j]
                        if new_total > current_sum:
                            selected.remove(i)
                            selected.add(j)
                            current_sum = new_total
                            improved = True
                            break
            if improved:
                break
        # Try adding one item (if possible)
        if not improved:
            for j in range(n):
                if j not in selected:
                    if current_sum + 1.0/denominators[j] < 1.0:
                        new_total = current_sum + 1.0/denominators[j]
                        if new_total > current_sum:
                            selected.add(j)
                            current_sum = new_total
                            improved = True
                            break
                            
    return list(selected)

