import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality.
    dp[k] is an integer where bit s is set if sum s is achievable by k elements.
    """
    n = len(elements)
    if n == 0:
        return [0] * 1
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for e in elements:
        # Update in reverse to avoid using the same element multiple times
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << e)
    return dp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    candidates = list(range(1, 513))
    
    orders = []
    for _ in range(10):
        c = candidates[:]
        random.shuffle(c)
        orders.append(c)
        
    orders.append(candidates[:])
    orders.append(candidates[::-1])
    
    alt = []
    l, h = 0, len(candidates) - 1
    while l <= h:
        alt.append(candidates[l])
        if l != h:
            alt.append(candidates[h])
        l += 1
        h -= 1
    orders.append(alt)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = []
        
        for x in order:
            if time.time() > deadline:
                break
            
            valid = True
            n = len(current_set)
            
            # Check 1: x is mean of subset of current_set
            if n >= 2:
                dp_curr = _build_dp(current_set)
                for k in range(2, n + 1):
                    target = k * x
                    # target can be large, but bit shift handles it.
                    # If target is negative, skip.
                    if target >= 0:
                        if (dp_curr[k] >> target) & 1:
                            valid = False
                            break
            
            if not valid:
                continue
                
            # Check 2: Any a in current_set becomes mean?
            # We check all a in current_set
            for a in current_set:
                others = [v for v in current_set if v != a]
                if len(others) == 0:
                    continue
                    
                # Build DP for others
                dp_others = _build_dp(others)
                
                # j ranges from 1 to len(others)
                # k = j + 1
                # Target sum for U (subset of others) is k*a - x
                for j in range(1, len(others) + 1):
                    k = j + 1
                    target_sum = k * a - x
                    if target_sum < 0:
                        continue
                    if (dp_others[j] >> target_sum) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(x)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
        if best_len >= 48:
            break
            
    return best_set
