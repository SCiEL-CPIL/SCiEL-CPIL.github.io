# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _check_addition_fast(current_set, dp_sums, x):
    """
    Check if adding x to current_set creates any violation.
    dp_sums[k] is the bitset of sums of subsets of size k from current_set.
    """
    n = len(current_set)
    
    # Check 1: Is x the mean of any subset of >=2 elements from current_set?
    # Subset S subset of current_set, |S|=k>=2, sum(S) = k*x
    # We have dp_sums[k] for current_set.
    for k in range(2, n + 1):
        target = k * x
        # If target is too large, it can't be in dp_sums[k]
        # Max sum of k elements is k * max(current_set)
        # But we can just check the bit
        if (dp_sums[k] >> target) & 1:
            return False
            
    # Check 2: For each a in current_set, is a the mean of some subset of new_set that includes x?
    # Subset S = {x} U T, T subset of current_set \ {a}, |T|=t>=1
    # sum(T) = a*(t+1) - x
    for a in current_set:
        # t=1: T={y}, y in current_set \ {a}, y = 2*a - x
        target_t1 = 2 * a - x
        if target_t1 in current_set and target_t1 != a:
            return False
        
        # t>=2: Need DP of current_set \ {a}
        # We can construct the DP for current_set \ {a} quickly?
        # Or just run a small DP.
        # Since n is small (<50), and we only do this if t=1 didn't catch it,
        # we can run a DP for current_set excluding a.
        
        # Optimization: Only check t up to a small limit first?
        # But we must be correct.
        
        # Let's run DP for others = current_set \ {a}
        others = [y for y in current_set if y != a]
        m = len(others)
        if m < 2:
            continue
        
        # We need to check if a*(t+1) - x is a sum of t elements from others
        # for t in 1..m
        # We already checked t=1.
        
        # Max target for t=m is a*(m+1) - x
        max_target = a * (m + 1) - x
        if max_target < 0:
            continue
        
        # If max_target is small, use bitset DP
        # If max_target is large, it's unlikely to be a sum of small elements?
        # Elements are <= 512.
        # Max sum of t elements is t * 512.
        # So if a*(t+1) - x > t * 512, it's impossible.
        
        mask = (1 << (min(max_target, 25000) + 1)) - 1
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for elem in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << elem)
                dp[k] &= mask
        
        for t in range(2, m + 1):
            target = a * (t + 1) - x
            if target <= min(max_target, 25000):
                if (dp[t] >> target) & 1:
                    return False
            # If target > 25000, it might still be possible if elements are large?
            # But elements <= 512, t <= 48. Max sum 48*512 = 24576.
            # So 25000 is a safe upper bound for any subset sum.
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    # Strategy 1: Greedy from low to high
    selected = []
    dp_sums = [0] * 50  # Max 48 elements
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        
        # Check addition
        if _check_addition_fast(selected, dp_sums, x):
            # Add x
            selected.append(x)
            # Update dp_sums
            n = len(selected)
            # dp_sums[k] for new set = old dp_sums[k] | (old dp_sums[k-1] << x)
            # We must iterate backwards
            for k in range(n, 0, -1):
                dp_sums[k] |= (dp_sums[k-1] << x)
            # Note: dp_sums[0] is always 1
            
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Greedy from high to low
    selected = []
    dp_sums = [0] * 50
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        
        if _check_addition_fast(selected, dp_sums, x):
            selected.append(x)
            n = len(selected)
            for k in range(n, 0, -1):
                dp_sums[k] |= (dp_sums[k-1] << x)
            
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 3: Randomized greedy with more seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = []
        dp_sums = [0] * 50
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            
            if _check_addition_fast(selected, dp_sums, x):
                selected.append(x)
                n = len(selected)
                for k in range(n, 0, -1):
                    dp_sums[k] |= (dp_sums[k-1] << x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Strategy 4: Try to add more elements to best_set
    if time.time() < deadline - 5:
        current = best_set[:]
        dp_sums = [0] * 50
        # Rebuild dp_sums for current
        for x in current:
            n = len(current)
            for k in range(n, 0, -1):
                dp_sums[k] |= (dp_sums[k-1] << x)
                
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            
            if _check_addition_fast(current, dp_sums, x):
                current.append(x)
                n = len(current)
                for k in range(n, 0, -1):
                    dp_sums[k] |= (dp_sums[k-1] << x)
                
        if len(current) > len(best_set):
            best_set = current[:]
            
    return best_set[:48]
# EVOLVE-BLOCK-END
