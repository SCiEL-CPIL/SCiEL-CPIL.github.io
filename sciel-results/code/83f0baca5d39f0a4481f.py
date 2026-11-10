# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _check_3ap_fast(current_set, x):
    """Fast check for 3-AP violations (subset size 2)."""
    # Check if x is mean of any pair in current_set
    # 2*x = a + b
    # Check if any a in current_set is mean of x and b in current_set
    # 2*a = x + b => b = 2*a - x
    
    set_curr = set(current_set)
    
    # 1. x is mean of a, b in current_set
    # a + b = 2x
    # Iterate a in current_set, check if 2x - a in set_curr and 2x - a != a
    for a in current_set:
        b = 2 * x - a
        if b in set_curr and b != a:
            return False
            
    # 2. a in current_set is mean of x, b in current_set
    # 2*a = x + b => b = 2*a - x
    # b must be in set_curr, b != x, b != a
    for a in current_set:
        b = 2 * a - x
        if b in set_curr and b != a and b != x:
            return False
            
    return True

def _is_valid_fast(set_vals):
    """Fast check: no element is the mean of any subset of >=2 other elements."""
    n = len(set_vals)
    if n < 3:
        return True
    arr = np.array(set_vals, dtype=np.int64)
    max_val = int(arr.max())
    
    for a in set_vals:
        others = [x for x in set_vals if x != a]
        m = len(others)
        if m < 2:
            continue
        
        max_target = m * a
        
        # Use list of ints
        dp = [0] * (m + 1)
        dp[0] = 1  # bit 0 set
        
        for x in others:
            # Update in reverse
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
                # Mask to avoid huge integers
                if max_target < 20000:
                    dp[k] &= (1 << (max_target + 1)) - 1
        
        # Check for violations
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if (dp[k] >> target) & 1:
                    return False
    
    return True

def _check_addition_fast(current_set, x):
    """Check if adding x to current_set creates any violation."""
    # Fast 3-AP check first
    if not _check_3ap_fast(current_set, x):
        return False
        
    new_set = current_set + [x]
    n = len(new_set)
    if n < 3:
        return True
    
    # Check 1: Is x the mean of any subset of >=2 elements from current_set?
    if len(current_set) >= 2:
        others = current_set
        m = len(others)
        max_target = m * x
        if max_target < 50000:
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
                    if max_target < 20000:
                        dp[k] &= (1 << (max_target + 1)) - 1
            for k in range(2, m + 1):
                target = k * x
                if target <= max_target and (dp[k] >> target) & 1:
                    return False
    
    # Check 2: For each a in current_set, is a the mean of some subset of new_set that includes x?
    for a in current_set:
        others_no_a = [y for y in current_set if y != a]
        m = len(others_no_a)
        if m < 1:
            continue
        max_target = a * (m + 1) - x
        if max_target < 0:
            continue
        if max_target > 50000:
            continue
        dp = [0] * (m + 1)
        dp[0] = 1
        for elem in others_no_a:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << elem)
                if max_target < 20000:
                    dp[k] &= (1 << (max_target + 1)) - 1
        for t in range(1, m + 1):
            target = a * (t + 1) - x
            if 0 <= target <= max_target:
                if (dp[t] >> target) & 1:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin before 90s kill
    
    best_set = []
    
    # Strategy 1: Greedy from low to high
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
        
    # Strategy 2: Greedy from high to low
    selected = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
        
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Strategy 4: Try to improve best_set by swapping
    if time.time() < deadline - 5:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                # Try replacing current[i] with some x
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current[:i] + [x] + current[i+1:]
                    if _is_valid_fast(candidate):
                        # Same size, but might enable further additions
                        for y in range(1, 513):
                            if y in candidate:
                                continue
                            if len(candidate) >= 48:
                                break
                            if _check_addition_fast(candidate, y):
                                candidate.append(y)
                                improved = True
                                break
                        current = candidate
                        if len(current) > len(best_set):
                            best_set = current[:]
                        break
                if improved:
                    break
                    
    # Strategy 5: Local search - try adding elements to best_set
    if time.time() < deadline - 3:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]
            
    return best_set[:48]
# EVOLVE-BLOCK-END
