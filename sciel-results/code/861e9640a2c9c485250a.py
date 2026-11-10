import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit
    
    # Try increasing k from minimum possible
    # Minimum k: sum of 1/5 + 1/6 + ... + 1/(5+k-1) >= 1
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 = 0.2+0.1667+0.1429+0.125+0.1111+0.1 = 0.8457
    # Need more terms. Let's just try k from 5 upward.
    
    for k in range(5, 50):
        if time.time() > deadline:
            break
        result = _dfs_egyptian(1, 5, k, [])
        if result is not None:
            return result
    # Fallback: use a known valid representation
    return _fallback()

def _dfs_egyptian(target, min_den, remaining, current):
    if remaining == 0:
        return current if target == 0 else None
    
    if target <= 0:
        return None
    
    # Lower bound: max sum with 'remaining' terms starting from min_den
    # is 1/min_den + 1/(min_den+1) + ... 
    # Upper bound check: min possible sum
    # If target > sum of 1/min_den + 1/(min_den+1) + ... + 1/(min_den+remaining-1), impossible
    max_sum = sum(Fraction(1, min_den + i) for i in range(remaining))
    if target > max_sum:
        return None
    
    # The next denominator d must satisfy:
    # 1/d >= target / remaining  =>  d <= remaining / target
    # Also d >= min_den
    # And we need: target - 1/d >= min possible sum of (remaining-1) terms from d+1
    max_d = min(10000000, int(remaining / target) + 1) if target > 0 else 10000000
    if max_d < min_den:
        return None
    
    for d in range(min_den, max_d + 1):
        new_target = target - Fraction(1, d)
        if new_target < 0:
            break
        # Check if new_target can be represented by (remaining-1) terms from d+1
        if new_target > 0:
            min_possible = sum(Fraction(1, d + 1 + i) for i in range(remaining - 1))
            if new_target > min_possible:
                continue
        result = _dfs_egyptian(new_target, d + 1, remaining - 1, current + [d])
        if result is not None:
            return result
    return None

def _fallback():
    # A valid representation: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Let me just use a brute force known one
    # 1 = 1/2 + 1/3 + 1/6 but 2,3,6 - 2 and 3 are < 5
    # Let's use: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Verify: this should be close to 1
    # Actually let me just return a simple valid one found by the original program
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no representation found")
