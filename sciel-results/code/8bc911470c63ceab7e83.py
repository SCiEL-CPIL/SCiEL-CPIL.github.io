"""Rewritten program using meet-in-the-middle with exact integer arithmetic."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Denominators 2..48, zero-based indices
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Compute LCM of 2..48
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = 1
    for d in denoms:
        L = lcm(L, d)
    
    # Convert each 1/d to integer weight = L/d
    weights = [L // d for d in denoms]
    
    # Split into two halves
    mid = n // 2  # 23
    left_weights = weights[:mid]   # 23 items
    right_weights = weights[mid:]  # 24 items
    
    # Enumerate all subset sums for left half
    # Store (sum, bitmask) for each subset
    left_sums = []
    num_left = 1 << mid
    for mask in range(num_left):
        s = 0
        for i in range(mid):
            if mask & (1 << i):
                s += left_weights[i]
        left_sums.append((s, mask))
    
    # Sort left sums by sum value
    left_sums.sort(key=lambda x: x[0])
    
    # Extract just the sums for binary search
    left_sum_values = [s for s, m in left_sums]
    
    import bisect
    
    best_total = 0
    best_left_mask = 0
    best_right_mask = 0
    
    # Enumerate all subset sums for right half
    num_right = 1 << (n - mid)
    for mask in range(num_right):
        s_right = 0
        for i in range(n - mid):
            if mask & (1 << i):
                s_right += right_weights[i]
        
        # We need s_left + s_right < L
        # So s_left < L - s_right
        target = L - s_right
        # Find the largest s_left < target
        idx = bisect.bisect_left(left_sum_values, target) - 1
        if idx >= 0:
            total = left_sum_values[idx] + s_right
            if total > best_total:
                best_total = total
                best_left_mask = left_sums[idx][1]
                best_right_mask = mask
    
    # Reconstruct indices
    result = []
    for i in range(mid):
        if best_left_mask & (1 << i):
            result.append(i)
    for i in range(n - mid):
        if best_right_mask & (1 << i):
            result.append(mid + i)
    
    # Ensure nonempty
    if not result:
        result = [0]  # index 0 corresponds to denominator 2, 1/2 < 1
    
    return result
# EVOLVE-BLOCK-END
