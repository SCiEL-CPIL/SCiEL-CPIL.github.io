import time
from math import gcd
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55  # 55s safety margin before 60s kill
    
    denominators = list(range(2, 49))  # 47 denominators
    n = len(denominators)
    
    # Compute LCM of all denominators
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
    
    # weights: weight of each denominator = L // d
    weights = [L // d for d in denominators]
    
    target = L  # We want sum < L (strictly below 1)
    
    # Sort by weight descending (largest weights first) for better pruning
    # Keep track of original indices
    order = sorted(range(n), key=lambda i: -weights[i])
    sorted_weights = [weights[i] for i in order]
    
    # Precompute suffix sums for upper bound
    # suffix[i] = sum of sorted_weights[i:]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sorted_weights[i]
    
    best_sum = -1
    best_mask = 0  # bitmask in sorted order
    
    # Branch and bound
    # At position pos, we've accumulated 'current_sum'
    # We can either take sorted_weights[pos] or skip it
    # Upper bound: current_sum + suffix[pos] (take all remaining)
    # Prune if current_sum + suffix[pos] <= best_sum (can't improve)
    
    def bbs(pos, current_sum, mask):
        nonlocal best_sum, best_mask
        
        if time.time() > deadline:
            return
        
        # Prune: even taking all remaining can't beat best
        if current_sum + suffix[pos] <= best_sum:
            return
        
        # Prune: already at or above target
        if current_sum >= target:
            return
        
        # Update best if current sum is valid (strictly below target)
        if current_sum > best_sum:
            best_sum = current_sum
            best_mask = mask
        
        # Base case
        if pos >= n:
            return
        
        # Option 1: take item pos
        new_sum = current_sum + sorted_weights[pos]
        if new_sum < target:
            bbs(pos + 1, new_sum, mask | (1 << pos))
        
        # Option 2: skip item pos
        bbs(pos + 1, current_sum, mask)
    
    bbs(0, 0, 0)
    
    # Convert best_mask (in sorted order) to original indices
    if best_sum < 0:
        # Fallback: just take the largest single weight below target
        for i in range(n):
            if weights[i] < target:
                return [i]
        return [0]
    
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(order[i])
    
    return sorted(result)

def greedy_fallback(denominators, L, target):
    """Simple greedy fallback."""
    selected = []
    current_sum = 0
    
    for d in sorted(denominators):
        w = L // d
        if current_sum + w < target:
            selected.append(d)
            current_sum += w
    
    result = [denominators.index(d) for d in selected]
    return result
