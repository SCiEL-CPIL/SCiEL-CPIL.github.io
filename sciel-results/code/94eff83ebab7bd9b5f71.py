"""Rewritten program: search for largest dissociated reciprocal subset of [1..22]."""
import time
from fractions import Fraction

_DEADLINE = time.time() + 80  # safety margin under 90s

def _check_add(sums_set, r):
    new_sums = [s + r for s in sums_set]
    for s in new_sums:
        if s in sums_set:
            return sums_set, False
    if len(new_sums) != len(set(new_sums)):
        return sums_set, False
    return sums_set | set(new_sums), True

def _is_dissociated(S):
    sums_set = {Fraction(0, 1)}
    for x in S:
        _, ok = _check_add(sums_set, Fraction(1, x))
        if not ok:
            return False
    return True

def _greedy_build(order, target_size):
    S = []
    sums_set = {Fraction(0, 1)}
    for x in order:
        if x in S:
            continue
        new_sums_set, ok = _check_add(sums_set, Fraction(1, x))
        if ok:
            S.append(x)
            sums_set = new_sums_set
            if len(S) >= target_size:
                break
    return S

def _local_grow(S):
    """Try to add any unused element; repeat until stuck. Returns (S, sums_set)."""
    S = list(S)
    sums_set = {Fraction(0, 1)}
    for x in S:
        sums_set, _ = _check_add(sums_set, Fraction(1, x))
    changed = True
    while changed and time.time() < _DEADLINE:
        changed = False
        for x in range(1, 23):
            if x in S:
                continue
            new_sums_set, ok = _check_add(sums_set, Fraction(1, x))
            if ok:
                S.append(x)
                sums_set = new_sums_set
                changed = True
                break
    return S, sums_set

def _try_size(k):
    universe = list(range(1, 23))
    orders = [
        sorted(universe, reverse=True),
        sorted(universe, key=lambda x: (x % 2, -x)),
        universe,
        sorted(universe, key=lambda x: -1/x),
        sorted(universe, reverse=True, key=lambda x: x % 3),
        sorted(universe, key=lambda x: (x % 4, -x)),
    ]
    for order in orders:
        if time.time() > _DEADLINE:
            break
        base = _greedy_build(order, k)
        grown, _ = _local_grow(base)
        if len(grown) >= k:
            return grown[:k] if len(grown) > k else grown
    # DFS fallback
    cand_order = sorted(universe, reverse=True)
    found = [None]
    def dfs(start, current, sums_set, remaining_to_pick):
        if time.time() > _DEADLINE:
            return
        if len(current) == k:
            found[0] = list(current)
            return
        idx = start
        n = len(cand_order)
        while idx < n:
            if time.time() > _DEADLINE:
                return
            x = cand_order[idx]
            if n - idx < remaining_to_pick:
                break
            new_sums_set, ok = _check_add(sums_set, Fraction(1, x))
            if ok:
                current.append(x)
                dfs(idx + 1, current, new_sums_set, remaining_to_pick - 1)
                current.pop()
                if found[0] is not None:
                    return
            idx += 1
    dfs(0, [], {Fraction(0, 1)}, k)
    if found[0] is not None:
        return found[0]
    return None

def construct():
    t0 = time.time()
    for k in range(22, 8, -1):
        if time.time() - t0 > 75:
            break
        res = _try_size(k)
        if res is not None:
            return res
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
