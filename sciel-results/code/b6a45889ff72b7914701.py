"""Rewritten: precomputed conflict-subset index + fast greedy + local search."""
import time
import random
from itertools import combinations

DEADLINE = 85.0  # safety margin under 90s wall-clock
MAX_VAL = 512


def _build_conflict_index():
    """conflict[x] = list of tuples (x, frozenset(others)) where mean(others)==x.
    Any selected set containing x and all of 'others' is invalid."""
    n = MAX_VAL
    conflict = [[] for _ in range(n + 1)]
    for k in range(2, 9):  # subset size of OTHERS (mean of >=2 others)
        target_sum = k * (n + 1) // 2
        for x in range(1, n + 1):
            need = k * x
            if need > target_sum:
                break
            if need < k:
                continue
            for combo in combinations(range(1, x), k):
                if sum(combo) == need:
                    conflict[x].append((x, frozenset(combo)))
    return conflict


def _is_valid_greedy(conflict, selected):
    """Check if `selected` is a valid set using the conflict index."""
    sel_set = set(selected)
    for x in selected:
        for _, others in conflict[x]:
            if others <= sel_set:
                return False
    return True


def _greedy_build(conflict, start, end, order, rng):
    """Greedy: add elements in given order if valid."""
    selected = []
    sel_set = set()
    for x in order:
        ok = True
        for _, others in conflict[x]:
            if others <= sel_set:
                ok = False
                break
        if ok:
            selected.append(x)
            sel_set.add(x)
    return selected


def _try_add(conflict, selected, sel_set, x):
    """Check if x can be added to selected without creating a violation."""
    for _, others in conflict[x]:
        if others <= sel_set:
            return False
    return True


def _local_search(conflict, selected, sel_set, unused, rng, deadline):
    """Try to improve by removing one and adding two, or swapping."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            removed = selected[i]
            new_sel = selected[:i] + selected[i+1:]
            new_set = sel_set - {removed}
            # try adding two unused elements
            for a in unused:
                if a == removed:
                    continue
                if not _try_add(conflict, new_sel, new_set, a):
                    continue
                set_a = new_set | {a}
                for b in unused:
                    if b == a or b == removed:
                        continue
                    if not _try_add(conflict, new_sel + [a], set_a, b):
                        continue
                    # found improvement: remove 1, add 2
                    new_sel2 = new_sel + [a, b]
                    new_sel2.sort()
                    selected = new_sel2
                    sel_set = set(new_sel2)
                    unused = [u for u in unused if u not in (a, b)]
                    if removed not in unused:
                        unused.append(removed)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
    return selected, sel_set, unused


def construct():
    t0 = time.time()
    conflict = _build_conflict_index()

    best = []
    best_size = 0

    # Multiple randomized greedy passes
    rng = random.Random(42)
    orderings = []

    # Ascending
    orderings.append(list(range(1, MAX_VAL + 1)))
    # Descending
    orderings.append(list(range(MAX_VAL, 0, -1)))
    # Random shuffles
    for _ in range(20):
        o = list(range(1, MAX_VAL + 1))
        rng.shuffle(o)
        orderings.append(o)
    # Biased: prefer middle values (fewer conflicts)
    for _ in range(10):
        o = list(range(1, MAX_VAL + 1))
        # Weight towards middle
        weights = [1.0 / (1.0 + abs(x - 256) / 100.0) for x in o]
        # Simple weighted shuffle
        for i in range(len(o) - 1, 0, -1):
            total = sum(weights[j] for j in range(i + 1))
            r = rng.random() * total
            acc = 0
            for j in range(i + 1):
                acc += weights[j]
                if acc >= r:
                    o[i], o[j] = o[j], o[i]
                    weights[i], weights[j] = weights[j], weights[i]
                    break
        orderings.append(o)

    for order in orderings:
        if time.time() > deadline:
            break
        sel = _greedy_build(conflict, 1, MAX_VAL, order, rng)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
            # Try local search on the best
            sel_set = set(sel)
            unused = [x for x in range(1, MAX_VAL + 1) if x not in sel_set]
            sel, sel_set, unused = _local_search(conflict, sel, sel_set, unused, rng, deadline)
            if len(sel) > best_size:
                best_size = len(sel)
                best = sel

    # Final local search pass on best if time remains
    if time.time() < deadline - 2:
        sel_set = set(best)
        unused = [x for x in range(1, MAX_VAL + 1) if x not in sel_set]
        best, sel_set, unused = _local_search(conflict, best, sel_set, unused, rng, deadline)

    # Cap at 48
    if len(best) > 48:
        best = best[:48]

    return best


deadline = DEADLINE
