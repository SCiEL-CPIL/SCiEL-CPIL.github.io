"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt
import time
from itertools import combinations

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _validate(arr):
    if not isinstance(arr, (list, tuple)):
        return False
    if not (1 <= len(arr) <= 100):
        return False
    seen = set()
    prod = P['n']
    for x in arr:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > 10210:
            return False
        if x in seen:
            return False
        seen.add(x)
        prod *= x
    s = isqrt(prod)
    return s * s == prod

def construct():
    start = time.time()
    deadline = 50.0
    
    # Known best: [224, 240] with max 240. Score 21.
    best = [224, 240]
    best_val = 240 - 210
    
    # Search for better solution (max < 240)
    # 1-element
    for x in range(211, 240):
        prod = 210 * x
        s = isqrt(prod)
        if s * s == prod:
            return [x]
            
    # 2-element
    nums = list(range(211, 240))
    for i in range(len(nums)):
        base = 210 * nums[i]
        for j in range(i+1, len(nums)):
            prod = base * nums[j]
            s = isqrt(prod)
            if s * s == prod:
                return [nums[i], nums[j]]
                
    # 3-element
    if time.time() - start < deadline:
        for c in combinations(nums, 3):
            prod = 210 * c[0] * c[1] * c[2]
            s = isqrt(prod)
            if s * s == prod:
                return list(c)
                
    # 4-element
    if time.time() - start < deadline:
        for c in combinations(nums, 4):
            prod = 210 * c[0] * c[1] * c[2] * c[3]
            s = isqrt(prod)
            if s * s == prod:
                return list(c)

    return best
# EVOLVE-BLOCK-END
