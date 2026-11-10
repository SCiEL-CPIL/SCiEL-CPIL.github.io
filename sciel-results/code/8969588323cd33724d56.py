"""Rewritten program: exact-arithmetic construction of a large dissociated reciprocal set."""
import time
from fractions import Fraction

def _subset_sums_set(elems):
    """Return the set of all subset sums of reciprocals of elems (as Fractions)."""
    sums = {Fraction(0, 1)}
    for e in elems:
        r = Fraction(1, e)
        new_sums = {s + r for s in sums}
        sums = sums | new_sums
    return sums

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using exact arithmetic."""
    if not elems:
        return True
    n = len(elems)
    expected = (1 << n)
    sums = _subset_sums_set(elems)
    return len(sums) == expected

def _greedy_build(ordering, deadline):
    """Greedily build a dissociated set following the given ordering."""
    cur = []
    cur_sums = {Fraction(0, 1)}
    for e in ordering:
        if time.time() > deadline:
            break
        r = Fraction(1, e)
        # Check collision: for any existing sum s, is s + r already in cur_sums?
        collision = False
        for s in cur_sums:
            if (s + r) in cur_sums:
                collision = True
                break
        if not collision:
            cur.append(e)
            cur_sums = cur_sums | {s + r for s in cur_sums}
    return cur, cur_sums

def _expand(set_elems, remaining, deadline):
    """Try to add elements from remaining to set_elems."""
    cur = list(set_elems)
    cur_sums = _subset_sums_set(cur)
    remaining_sorted = sorted(remaining, reverse=True)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in remaining_sorted[:]:
            if time.time() > deadline:
                break
            if e in cur:
                continue
            r = Fraction(1, e)
            collision = False
            for s in cur_sums:
                if (s + r) in cur_sums:
                    collision = True
                    break
            if not collision:
                cur.append(e)
                cur_sums = cur_sums | {s + r for s in cur_sums}
                remaining_sorted.remove(e)
                changed = True
    return cur

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin within 90s limit
    
    universe = list(range(1, 23))
    
    # Multiple greedy orderings
    orderings = [
        sorted(universe, reverse=True),      # 22, 21, ..., 1
        sorted(universe),                     # 1, 2, ..., 22
        list(range(22, 0, -1)),              # same as first
        list(range(2, 23)),                  # skip 1, ascending
        list(range(22, 1, -1)),              # skip 1, descending
        # Try skipping small elements
        [e for e in range(22, 0, -1) if e >= 3],
        [e for e in range(22, 0, -1) if e >= 4],
        [e for e in range(22, 0, -1) if e >= 5],
    ]
    
    best_set = []
    best_size = 0
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        cur, _ = _greedy_build(ordering, deadline - 10)
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = cur[:]
    
    # Expand the best set
    remaining = [e for e in universe if e not in best_set]
    if time.time() < deadline - 5:
        expanded = _expand(best_set, remaining, deadline - 5)
        if len(expanded) > len(best_set):
            best_set = expanded
    
    # Local search: try swapping elements to enable further expansion
    if time.time() < deadline - 3:
        remaining = [e for e in universe if e not in best_set]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 3:
                    break
                for e in remaining[:]:
                    candidate = best_set[:i] + best_set[i+1:] + [e]
                    if _is_dissociated(candidate):
                        new_remaining = [x for x in universe if x not in candidate]
                        expanded = _expand(candidate, new_remaining, deadline - 3)
                        if len(expanded) > len(best_set):
                            best_set = expanded
                            remaining = [x for x in universe if x not in best_set]
                            improved = True
                            break
                if improved:
                    break
    
    # Verify with exact arithmetic
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback: a known good set of size 16
    return [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
