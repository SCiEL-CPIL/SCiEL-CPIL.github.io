"""Find largest dissociated reciprocal subset of [1,22]."""
import time
import random
from math import lcm
from itertools import combinations

# Deadline guard: 80s of 90s wall-clock limit
_DEADLINE = 80.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _check_dissociated_lcm(scaled_weights):
    """Check if all 2^n subset sums of scaled_weights are distinct."""
    n = len(scaled_weights)
    if n == 0:
        return True
    # Use set of subset sums
    sums = {0}
    for w in scaled_weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _check_dissociated_mod(scaled_weights, mod):
    """Necessary condition: all subset sums distinct mod P."""
    n = len(scaled_weights)
    if n == 0:
        return True
    sums = {0}
    for w in scaled_weights:
        wm = w % mod
        new_sums = set()
        for s in sums:
            ns = (s + wm) % mod
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _greedy_construct(order, deadline):
    """Greedy: try to add elements in given order, checking dissociation."""
    selected = []
    # LCM of selected elements' reciprocals = LCM of selected elements
    # Scaled weights: for element i, weight = LCM / i
    # We rebuild LCM as we go
    
    for idx in order:
        if _time_left() < 2.0:
            break
        candidate = selected + [idx]
        # Compute LCM of candidate
        l = 1
        for x in candidate:
            l = lcm(l, x)
        if l > 10**18:  # Too large, skip
            continue
        # Check modular filter first (fast)
        P = 10**9 + 7
        weights_mod = [l // x % P for x in candidate]
        if not _check_dissociated_mod(weights_mod, P):
            continue
        # Full check
        weights = [l // x for x in candidate]
        if _check_dissociated_lcm(weights):
            selected = candidate
    return selected

def _random_order():
    """Generate a random ordering of [1..22]."""
    lst = list(range(1, 23))
    random.shuffle(lst)
    return lst

def _local_search(current, deadline):
    """Try to improve by swapping elements."""
    if not current:
        return current
    current = list(current)
    available = [x for x in range(1, 23) if x not in current]
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(len(current)):
            for a in available:
                candidate = current[:i] + [a] + current[i+1:]
                l = 1
                for x in candidate:
                    l = lcm(l, x)
                if l > 10**18:
                    continue
                P = 10**9 + 7
                weights_mod = [l // x % P for x in candidate]
                if not _check_dissociated_mod(weights_mod, P):
                    continue
                weights = [l // x for x in candidate]
                if _check_dissociated_lcm(weights):
                    current = candidate
                    available = [x for x in range(1, 23) if x not in current]
                    improved = True
                    break
            if improved:
                break
    return current

def _try_add(current, deadline):
    """Try to add one more element to current set."""
    if not current:
        return current
    current = list(current)
    available = [x for x in range(1, 23) if x not in current]
    random.shuffle(available)
    for a in available:
        if _time_left() < 3.0:
            break
        candidate = current + [a]
        l = 1
        for x in candidate:
            l = lcm(l, x)
        if l > 10**18:
            continue
        P = 10**9 + 7
        weights_mod = [l // x % P for x in candidate]
        if not _check_dissociated_mod(weights_mod, P):
            continue
        weights = [l // x for x in candidate]
        if _check_dissociated_lcm(weights):
            return candidate
    return current

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # seed fallback, size 9
    
    # Multiple greedy attempts with different orderings
    attempts = 0
    while _time_left() > 10.0:
        order = _random_order()
        result = _greedy_construct(order, _DEADLINE)
        if len(result) > len(best):
            best = result
        attempts += 1
        if attempts > 200 and _time_left() > 5.0:
            break
    
    # Try to extend best
    if _time_left() > 5.0:
        extended = _try_add(best, _DEADLINE)
        if len(extended) > len(best):
            best = extended
    
    # Local search
    if _time_left() > 5.0:
        improved = _local_search(best, _DEADLINE)
        if len(improved) > len(best):
            best = improved
    
    # Final attempt to add
    if _time_left() > 3.0:
        extended = _try_add(best, _DEADLINE)
        if len(extended) > len(best):
            best = extended
    
    return best
