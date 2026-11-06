from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def is_valid(S):
    S = list(set(S))
    if not S:
        return False
    M = reduce(_lcm, S)
    for a in S:
        target = M // a
        items = [M // v for v in S if v != a]
        # Filter items > target
        items = [x for x in items if x <= target]
        if not items:
            continue
        # Subset sum check
        reachable = {0}
        for item in items:
            new_reachable = set()
            for s in reachable:
                ns = s + item
                if ns <= target:
                    new_reachable.add(ns)
            reachable |= new_reachable
            if target in reachable:
                return False
    return True
