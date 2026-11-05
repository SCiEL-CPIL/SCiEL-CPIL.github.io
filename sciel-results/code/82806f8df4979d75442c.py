"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START

_LCM = 5354228880

def _recip_int(n):
    return _LCM // n

def _can_form_target(target, values):
    """Check if target can be formed as sum of distinct values from values (non-empty subset)."""
    if not values:
        return False
    if target <= 0:
        return False
    # Early check: if sum of all values < target, impossible
    if sum(values) < target:
        return False
    # If any single value equals target
    for v in values:
        if v == target:
            return True
    # Subset sum using set of reachable sums
    reachable = {0}
    for v in values:
        new_reachable = set()
        for s in reachable:
            ns = s + v
            if ns == target:
                return True
            if ns < target:  # Prune: all values positive, no need to exceed target
                new_reachable.add(ns)
        reachable |= new_reachable
    return False

def _is_valid_set(S):
    """Check if set S is reciprocal-subsum-free."""
    S_list = sorted(S)
    n = len(S_list)
    if n <= 1:
        return True
    
    for i, a in enumerate(S_list):
        target = _LCM // a
        others = [_LCM // b for j, b in enumerate(S_list) if j != i]
        if _can_form_target(target, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_set = set(range(13, 25))
    best_size = len(best_set)
    
    def try_extend(base, candidates, order=None):
        nonlocal best_set, best_size
        if order is None:
            order = candidates
        current = set(base)
        for c in order:
            if time.time() > deadline:
                return
            if c in current:
                continue
            test = current | {c}
            if _is_valid_set(test):
                current = test
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Strategy 1: Extend from high range with low elements
    try_extend(set(range(13, 25)), list(range(1, 13)))
    try_extend(set(range(13, 25)), list(range(12, 0, -1)))
    
    # Strategy 2: Different base sets
    for bs in range(10, 18):
        if time.time() > deadline:
            break
        base = set(range(bs, 25))
        if _is_valid_set(base):
            try_extend(base, list(range(1, bs)))
    
    # Strategy 3: Greedy high-to-low
    if time.time() < deadline:
        current = set()
        for x in range(24, 0, -1):
            if time.time() > deadline:
                break
            test = current | {x}
            if _is_valid_set(test):
                current = test
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Strategy 4: Greedy low-to-high
    if time.time() < deadline:
        current = set()
        for x in range(1, 25):
            if time.time() > deadline:
                break
            test = current | {x}
            if _is_valid_set(test):
                current = test
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Strategy 5: Random orderings
    if time.time() < deadline - 5.0:
        rng = random.Random(999)
        for _ in range(200):
            if time.time() > deadline:
                break
            order = list(range(1, 25))
            rng.shuffle(order)
            current = set()
            for x in order:
                test = current | {x}
                if _is_valid_set(test):
                    current = test
            if len(current) > best_size:
                best_set = current
                best_size = len(current)
    
    # Strategy 6: Local search swaps
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for a in list(best_set):
                if time.time() > deadline:
                    break
                for b in range(1, 25):
                    if b in best_set:
                        continue
                    test = (best_set - {a}) | {b}
                    if _is_valid_set(test):
                        best_set = test
                        improved = True
                        break
                if improved:
                    break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
