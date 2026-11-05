import time
import random

def _subset_sum_exists(values, target, min_size):
    """
    Check if there is a subset of 'values' with sum == target and size >= min_size.
    values: list of integers (can be negative).
    target: integer.
    min_size: integer.
    Returns True if such a subset exists.
    """
    n = len(values)
    if n < min_size:
        return False
    
    # Find min and max possible sum to set offset
    min_sum = sum(v for v in values if v < 0)
    max_sum = sum(v for v in values if v > 0)
    
    # If target is out of range, return False
    if target < min_sum or target > max_sum:
        return False
        
    offset = -min_sum
    # Bitset size
    size = max_sum - min_sum + 1
    # We need to track sizes? 
    # Standard subset sum bitset dp[k] where bit i is set if sum i-offset is achievable with k elements.
    # But n is small (<=48). We can do dp[k] as a bitmask.
    
    # Initialize dp
    # dp[k] is an integer bitmask. Bit j is set if sum j-offset is achievable with k elements.
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # Sum 0 with 0 elements
    
    for v in values:
        # Update backwards
        # We need to shift the bitsets
        # If v >= 0, shift left by v. If v < 0, shift right by -v.
        # But we also need to handle the offset correctly.
        # Actually, the bitset represents sums.
        # Let's just use a list of sets of achievable sums for each size? 
        # n is small, number of sums is bounded.
        
        # Alternative: Use a 2D array of booleans? 
        # dp[k][s] = True if sum s is achievable with k elements.
        # Range of s is [min_sum, max_sum].
        # This is O(n * range). Range ~ 50000. n ~ 48. 48 * 50000 = 2.4M operations. 
        # This is fast in Python if optimized, but bitset is faster.
        
        # Let's stick to bitset per size.
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= shifted
            
    # Check if any k >= min_size has the target bit set
    target_bit = 1 << (target + offset)
    for k in range(min_size, n + 1):
        if dp[k] & target_bit:
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    
    # We will try multiple random orders and greedy construction
    # To maximize cardinality, we want to pick numbers that are "far apart" or "structurally independent".
    # Powers of 2 are good. Random numbers are good.
    
    candidates = list(range(1, 513))
    
    # Try a few specific strategies
    strategies = []
    
    # 1. Random shuffles
    for _ in range(20):
        random.shuffle(candidates)
        strategies.append(candidates[:])
        
    # 2. Powers of 2 and multiples
    # 3. Sorted orders (might get stuck in local optima but fast)
    strategies.append(list(range(1, 513)))
    strategies.append(list(range(512, 0, -1)))
    
    for order in strategies:
        if time.time() > deadline:
            break
            
        current_set = []
        for x in order:
            if time.time() > deadline:
                break
                
            # Check if x can be added
            if len(current_set) >= 48:
                break
                
            valid = True
            
            # Check 1: Is x the mean of any subset of current_set (size >= 2)?
            # Subset V of current_set, |V| >= 2, sum(v - x) = 0
            if len(current_set) >= 2:
                diffs = [v - x for v in current_set]
                if _subset_sum_exists(diffs, 0, 2):
                    valid = False
            
            # Check 2: Does adding x cause any a in current_set to become a mean?
            # For each a in current_set, check if there is V subset of current_set\{a} 
            # such that sum(v - a) = a - x and |V| >= 1.
            if valid:
                for i, a in enumerate(current_set):
                    # V is subset of current_set excluding a
                    others = [v for j, v in enumerate(current_set) if j != i]
                    if len(others) < 1:
                        continue
                    diffs = [v - a for v in others]
                    target = a - x
                    if _subset_sum_exists(diffs, target, 1):
                        valid = False
                        break
                        
            if valid:
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            if len(best_set) == 48:
                return best_set
                
    return best_set
