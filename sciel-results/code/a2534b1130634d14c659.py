import time
import random

_LCM_22 = 232792560

def _greedy_with_order(order):
    selected = []
    sums = {0}
    for i in order:
        w = _LCM_22 // i
        new_sums = set()
        ok = True
        for s in sums:
            ns = s + w
            if ns in sums:
                ok = False
                break
            new_sums.add(ns)
        if ok:
            sums.update(new_sums)
            selected.append(i)
    return selected

def construct():
    deadline = time.time() + 85.0
    all_elements = list(range(1, 23))
    
    best = []
    
    # Deterministic orderings
    orderings = [
        list(reversed(all_elements)),
        all_elements,
    ]
    
    rng = random.Random(12345)
    
    for order in orderings:
        result = _greedy_with_order(order)
        if len(result) > len(best):
            best = result
    
    # Random orderings
    while time.time() < deadline - 2.0:
        order = all_elements[:]
        rng.shuffle(order)
        result = _greedy_with_order(order)
        if len(result) > len(best):
            best = result
    
    if not best:
        return [1]
    return best
