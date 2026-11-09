"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

# Precompute LCM of 1..22
_LCM = 1
for i in range(1, 23):
    _LCM = _lcm(_LCM, i)

# Precompute integer representations of reciprocals
_INV = [0] + [_LCM // i for i in range(1, 23)]

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct using integer arithmetic."""
    sums = [0]
    for x in subset:
        inv = _INV[x]
        sums = sums + [s + inv for s in sums]
    return len(sums) == len(set(sums))

def construct():
    deadline = time.time() + 82.0
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best)
    
    # Strategy 1: Greedy from large to small
    current = []
    for x in range(22, 0, -1):
        if time.time() > deadline:
            break
        test = current + [x]
        if _is_dissociated(test):
            current = test
    if len(current) > best_size:
        best = current
        best_size = len(current)
    
    # Strategy 2: Greedy from small to large
    current = []
    for x in range(1, 23):
        if time.time() > deadline:
            break
        test = current + [x]
        if _is_dissociated(test):
            current = test
    if len(current) > best_size:
        best = current
        best_size = len(current)
    
    # Strategy 3: Add elements to best
    current = list(best)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 23):
            if time.time() > deadline:
                break
            if x not in current:
                test = current + [x]
                if _is_dissociated(test):
                    current = test
                    changed = True
                    if len(current) > best_size:
                        best = current
                        best_size = len(current)
    
    # Strategy 4: Local search with swaps + additions
    for iteration in range(30):
        if time.time() > deadline:
            break
        current = list(best)
        improved = False
        for i in range(len(current)):
            for x in range(1, 23):
                if time.time() > deadline:
                    break
                if x in current:
                    continue
                test = current[:i] + [x] + current[i+1:]
                if _is_dissociated(test):
                    # Try to add more
                    for y in range(1, 23):
                        if time.time() > deadline:
                            break
                        if y not in test:
                            test2 = test + [y]
                            if _is_dissociated(test2):
                                test = test2
                    if len(test) > best_size:
                        best = test
                        best_size = len(test)
                        improved = True
                        current = list(test)
                        break
            if time.time() > deadline:
                break
        if not improved:
            break
    
    # Strategy 5: Random greedy restarts
    random.seed(999)
    for attempt in range(1000):
        if time.time() > deadline:
            break
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        current = []
        for x in candidates:
            if time.time() > deadline:
                break
            test = current + [x]
            if _is_dissociated(test):
                current = test
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    # Final: try to add any remaining
    current = list(best)
    for x in range(1, 23):
        if time.time() > deadline:
            break
        if x not in current:
            test = current + [x]
            if _is_dissociated(test):
                current = test
                if len(current) > best_size:
                    best = current
                    best_size = len(current)
    
    return sorted(best)
# EVOLVE-BLOCK-END

