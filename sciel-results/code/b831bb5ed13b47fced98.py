"""Minimum maximal no-isosceles set: randomized greedy + local search (removal)."""
import time
import itertools
import numpy as np

N = 16
DEADLINE = time.time() + 320.0  # safety margin before 330s kill


def _can_add(pt, chosen):
    """Return True if adding pt to chosen keeps the set no-isosceles."""
    if len(chosen) < 2:
        return True
    x = np.array(chosen)
    d1 = (x[:, 0] - pt[0]) ** 2 + (x[:, 1] - pt[1]) ** 2
    d2 = (x[:, 0] - x[:, 0]) ** 2 + (x[:, 1] - x[:, 1]) ** 2  # placeholder
    # distances between pt and each chosen point
    # For each pair (a, b) in chosen, check if (pt, a, b) is isosceles.
    # Equivalent: for each a in chosen, count how many b in chosen have
    # dist(pt,a) == dist(a,b) or dist(pt,a) == dist(pt,b) or dist(a,b) == dist(pt,b).
    # Simpler: check each pair.
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            a = chosen[i]
            b = chosen[j]
            d1 = (pt[0] - a[0]) ** 2 + (pt[1] - a[1]) ** 2
            d2 = (pt[0] - b[0]) ** 2 + (pt[1] - b[1]) ** 2
            d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True


def _is_maximal(chosen):
    """Return (True, None) if chosen is maximal, else (False, removable_index)."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        # Check if p can be added: no isosceles with any pair in chosen
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                a = chosen[i]
                b = chosen[j]
                d1 = (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
                d2 = (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2
                d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return (False, p)
    return (True, None)


def _removal_local_search(chosen):
    """Try to remove points that preserve maximality. Return new list."""
    chosen = list(chosen)
    improved = True
    while improved:
        improved = False
        for i in range(len(chosen)):
            candidate = chosen[:i] + chosen[i + 1:]
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen


def _greedy_random():
    """Build a maximal no-isosceles set via randomized greedy."""
    points = list(itertools.product(range(N), repeat=2))
    np.random.shuffle(points)
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def construct():
    best = None
    best_len = float('inf')
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        # Randomized greedy construction
        chosen = _greedy_random()
        # Local search: try removals
        chosen = _removal_local_search(chosen)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = list(chosen)
        # Early exit if we find a very good solution
        if best_len <= 20:
            break
    if best is None:
        # Fallback: deterministic greedy
        chosen = []
        for p in itertools.product(range(N), repeat=2):
            if _can_add(p, chosen):
                chosen.append(p)
        return chosen
    return best
