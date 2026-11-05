import time
import random
import sys
from itertools import combinations
import numpy as np

_DEADLINE = 85.0

def _check_violation(C, idx):
    """
    Check if C[idx] is the mean of any subset of >=2 OTHER elements.
    Uses bitset DP.
    """
    a = C[idx]
    n = len(C)
    if n < 3:
        return False
    
    diffs = [C[j] - a for j in range(n) if j != idx]
    m = len(diffs)
    if m < 2:
        return False
    
    # Quick check for size 2
    diff_set = set(diffs)
    for d in diffs:
        if -d in diff_set and (-d != d or diffs.count(d) >= 2):
            return True
    
    # Quick check for size 3
    for i in range(m):
        for j in range(i+1, m):
            target = -diffs[i] - diffs[j]
            if target in diff_set:
                # Need to verify it's a distinct element
                count = 0
                for k in range(m):
                    if k != i and k != j and diffs[k] == target:
                        count += 1
                        break
                if count > 0:
                    return True
    
    # Calculate bounds for bitset DP
    max_pos = sum(d for d in diffs if d > 0)
    max_neg = -sum(d for d in diffs if d < 0)
    offset = max_neg
    total_span = max_pos + max_neg + 1
    
    if total_span > 100000:
        # Check size 4 with optimized approach
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    target = -diffs[i] - diffs[j] - diffs[k]
                    if target in diff_set:
                        for l in range(m):
                            if l != i and l != j and l != k and diffs[l] == target:
                                return True
        return False
        
    target_bit = 1 << offset
    mask = (1 << total_span) - 1
    
    dp = [0] * (m + 1)
    dp[0] = target_bit
    
    for d in diffs:
        for k in range(m, 0, -1):
            if dp[k-1] == 0:
                continue
            if d >= 0:
                dp[k] |= (dp[k-1] << d) & mask
            else:
                dp[k] |= (dp[k-1] >> (-d)) & mask
            if dp[k] & target_bit:
                return True
                
    return False

def _is_valid(C):
    for i in range(len(C)):
        if _check_violation(C, i):
            return False
    return True

def _subset_sum_bitset(arr, k, target):
    """Fast subset sum check using bitset for small k."""
    if k == 0:
        return target == 0
    if k > len(arr):
        return False
    if target < 0:
        return False
    
    n = len(arr)
    if k > 0:
        min_possible = sum(sorted(arr)[:k])
        max_possible = sum(sorted(arr)[-k:])
        if target < min_possible or target > max_possible:
            return False
    
    # Bitset DP: dp[c] is a bitset where bit s is set if sum s is achievable with c elements
    # We only care about sums up to target
    max_sum = target
    dp = [0] * (k + 1)
    dp[0] = 1  # bit 0 set
    
    for x in arr:
        if x > max_sum:
            continue
        for c in range(min(k, n), 0, -1):
            if dp[c-1] == 0:
                continue
            # Shift left by x, mask to max_sum bits
            shifted = (dp[c-1] << x) & ((1 << (max_sum + 1)) - 1)
            dp[c] |= shifted
            if dp[c] & (1 << target):
                return True
    
    return bool(dp[k] & (1 << target))

def _fast_incremental_check(C, new_elem):
    """
    Fast check for whether adding new_elem to C creates any violation.
    """
    a = new_elem
    n = len(C)
    
    # Check if new_elem is mean of any subset of existing elements
    for k in range(2, min(n, 6) + 1):
        target_sum = k * a
        if _subset_sum_bitset(C, k, target_sum):
            return False
    
    # Check if any C[i] is mean of subset containing new_elem
    for i in range(n):
        ci = C[i]
        others = [C[j] for j in range(n) if j != i]
        for k in range(2, min(len(others) + 1, 6) + 1):
            target_sum = k * ci - new_elem
            if target_sum < 0 or target_sum > sum(others):
                continue
            if _subset_sum_bitset(others, k - 1, target_sum):
                return False
    
    return True

def _greedy_construct(order, time_limit):
    selected = []
    start = time.time()
    
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        
        if _fast_incremental_check(selected, x):
            C = selected + [x]
            if _is_valid(C):
                selected.append(x)
    
    return selected

def _local_search_improve(C, time_limit):
    """
    Try to improve set C by removing some elements and adding others.
    """
    start = time.time()
    C = list(C)
    C_set = set(C)
    all_vals = set(range(1, 513))
    candidates = sorted(list(all_vals - C_set))
    
    while time.time() - start < time_limit and len(C) < 48:
        found = False
        
        # Try adding 1 element
        for x in candidates:
            if time.time() - start > time_limit:
                break
            if _fast_incremental_check(C, x):
                C_new = C + [x]
                if _is_valid(C_new):
                    C = C_new
                    C_set = set(C)
                    candidates = sorted(list(all_vals - C_set))
                    found = True
                    break
        
        if not found:
            # Try removing 1 and adding 1
            for i in range(len(C)):
                if time.time() - start > time_limit:
                    break
                C_temp = C[:i] + C[i+1:]
                C_temp_set = set(C_temp)
                for x in candidates:
                    if time.time() - start > time_limit:
                        break
                    if x in C_temp_set:
                        continue
                    if _fast_incremental_check(C_temp, x):
                        C_new = C_temp + [x]
                        if _is_valid(C_new):
                            C = C_new
                            C_set = set(C)
                            candidates = sorted(list(all_vals - C_set))
                            found = True
                            break
                if found:
                    break
        
        if not found:
            break
    
    return C

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Pre-defined good orders
    orders = []
    
    # 1. Lo-Hi alternating
    lo, hi = 1, 512
    oh = []
    while lo <= hi:
        oh.append(lo)
        if lo != hi: oh.append(hi)
        lo += 1
        hi -= 1
    orders.append(oh)
    
    # 2. Hi-Lo alternating
    lo, hi = 1, 512
    oh = []
    while lo <= hi:
        oh.append(hi)
        if lo != hi: oh.append(lo)
        lo += 1
        hi -= 1
    orders.append(oh)
    
    # 3. Random orders
    for seed in range(30):
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        orders.append(order)
        
    # 4. Spread order
    base = list(range(1, 513, 8))
    random.Random(123).shuffle(base)
    remaining = [x for x in range(1, 513) if x not in set(base)]
    random.Random(456).shuffle(remaining)
    orders.append(base + remaining)
    
    # 5. Powers of 2 based
    powers = [2**i for i in range(10) if 2**i <= 512]
    random.Random(789).shuffle(powers)
    rest = [x for x in range(1, 513) if x not in set(powers)]
    random.Random(999).shuffle(rest)
    orders.append(powers + rest)
    
    # Run greedy
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 15:
            break
            
        time_limit = min(1.5, _DEADLINE - elapsed - 10)
        if time_limit < 0.5:
            break
        result = _greedy_construct(order, time_limit)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
        if best_size == 48:
            break
            
    # Local search improvement
    if best_size < 48 and time.time() - start_time < _DEADLINE - 20:
        time_limit = min(15.0, _DEADLINE - time.time() - 10)
        if time_limit > 2:
            improved = _local_search_improve(best_set, time_limit)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # If we have more time, try more random orders
    if best_size < 48 and time.time() - start_time < _DEADLINE - 10:
        for seed in range(100, 200):
            if time.time() - start_time > _DEADLINE - 5:
                break
            rng = random.Random(seed)
            order = list(range(1, 513))
            rng.shuffle(order)
            result = _greedy_construct(order, 0.8)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                if best_size == 48:
                    break
    
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
