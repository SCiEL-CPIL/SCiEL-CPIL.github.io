import time
import random
import numpy as np

def _subset_sum_zero_bitset(diffs, min_size=2):
    """Check if there's a subset of diffs of size >= min_size that sums to 0 using bitset DP."""
    m = len(diffs)
    if m < min_size:
        return False
    
    # Compute bounds
    max_abs_sum = sum(abs(d) for d in diffs)
    if max_abs_sum == 0:
        return m >= min_size  # All zeros
    
    offset = max_abs_sum
    size = 2 * max_abs_sum + 1
    
    # dp[k] is a bitset of size 'size', where bit i is set if sum (i - offset) is achievable with k elements
    # We'll use numpy arrays of uint8 for efficiency
    # But numpy doesn't have efficient bitset operations for arbitrary size
    
    # Alternative: use Python integers as bitsets
    # dp[k] is a Python int where bit i represents sum (i - offset)
    
    # Initialize
    dp = [0] * (m + 1)
    dp[0] = 1 << offset  # sum 0 with 0 elements
    
    for d in diffs:
        shift = d
        for k in range(m - 1, -1, -1):
            if dp[k]:
                # Shift dp[k] by d and OR into dp[k+1]
                # But we need to handle bounds
                if shift >= 0:
                    new_bits = (dp[k] << shift) & ((1 << size) - 1)
                else:
                    new_bits = (dp[k] >> (-shift)) & ((1 << size) - 1)
                dp[k + 1] |= new_bits
                # Early termination
                if k + 1 >= min_size:
                    # Check if bit 'offset' (sum 0) is set
                    if (dp[k + 1] >> offset) & 1:
                        return True
    
    for k in range(min_size, m + 1):
        if (dp[k] >> offset) & 1:
            return True
    return False

def _subset_sum_target_bitset(diffs, target, min_size=1):
    """Check if there's a subset of diffs of size >= min_size that sums to target."""
    m = len(diffs)
    if m < min_size:
        return False
    
    # If target is 0 and min_size is 0, empty set works
    if target == 0 and min_size == 0:
        return True
    
    # Compute bounds
    max_abs_sum = sum(abs(d) for d in diffs)
    if max_abs_sum == 0:
        # All diffs are 0
        # Sum is always 0
        # Need subset of size >= min_size summing to target
        # If target == 0, need m >= min_size
        # Else, impossible
        return target == 0 and m >= min_size
    
    offset = max_abs_sum
    size = 2 * max_abs_sum + 1
    
    # Check if target is in range
    target_idx = target + offset
    if target_idx < 0 or target_idx >= size:
        return False
    
    dp = [0] * (m + 1)
    dp[0] = 1 << offset
    
    for d in diffs:
        shift = d
        for k in range(m - 1, -1, -1):
            if dp[k]:
                if shift >= 0:
                    new_bits = (dp[k] << shift) & ((1 << size) - 1)
                else:
                    new_bits = (dp[k] >> (-shift)) & ((1 << size) - 1)
                dp[k + 1] |= new_bits
                if k + 1 >= min_size:
                    if (dp[k + 1] >> target_idx) & 1:
                        return True
    
    for k in range(min_size, m + 1):
        if (dp[k] >> target_idx) & 1:
            return True
    return False

def _can_add(S, cand):
    """Check if cand can be added to S."""
    # Check 1: Is cand the mean of any subset of S with size >= 2?
    if len(S) >= 2:
        diffs = [x - cand for x in S]
        if _subset_sum_zero_bitset(diffs, min_size=2):
            return False
    
    # Check 2: For each a in S, is a the mean of a subset of S ∪ {cand} that includes cand?
    for a in S:
        target = a - cand
        U = [x for x in S if x != a]
        if not U:
            continue
        diffs_a = [x - a for x in U]
        if _subset_sum_target_bitset(diffs_a, target, min_size=1):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    best_set = []
    
    def greedy_build(order, current_set=None, max_time=None):
        S = list(current_set) if current_set else []
        S_set = set(S)
        for x in order:
            if time.time() > (max_time or deadline):
                break
            if x in S_set:
                continue
            if _can_add(S, x):
                S.append(x)
                S_set.add(x)
        return S
    
    all_nums = list(range(1, 513))
    
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        
        if attempt == 0:
            order = all_nums[:]
        elif attempt == 1:
            order = all_nums[::-1]
        elif attempt % 2 == 0:
            order = all_nums[:]
            random.shuffle(order)
        else:
            order = sorted(all_nums, key=lambda x: random.random() * x)
        
        S = greedy_build(order)
        
        if len(S) > len(best_set):
            best_set = S
    
    # Extend
    if best_set:
        S_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in S_set]
        
        for attempt in range(5):
            if time.time() > deadline - 3.0:
                break
            
            order = remaining[:]
            random.shuffle(order)
            
            S = list(best_set)
            S_set_local = set(S)
            for x in order:
                if time.time() > deadline - 2.0:
                    break
                if x in S_set_local:
                    continue
                if _can_add(S, x):
                    S.append(x)
                    S_set_local.add(x)
            
            if len(S) > len(best_set):
                best_set = S
    
    return best_set
