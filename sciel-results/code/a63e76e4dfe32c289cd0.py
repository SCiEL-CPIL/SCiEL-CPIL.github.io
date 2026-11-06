"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

DEADLINE = 320.0  # safety margin before 330s kill

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(p, S):
    """Check if adding p to S creates any isosceles triangle."""
    for i, a in enumerate(S):
        d1 = _dist2(p, a)
        for b in S[i+1:]:
            d2 = _dist2(p, b)
            d3 = _dist2(a, b)
            if d1 == d2 or d1 == d3 or d2 == d3:
                return True
    return False

def _is_maximal(S, all_points):
    """Check if S is maximal: no point outside S can be added without creating isosceles."""
    S_set = set(S)
    for p in all_points:
        if p in S_set:
            continue
        if not _is_isosceles(p, S):
            return False
    return True

def _greedy_construct(order, all_points):
    """Greedy: add points in given order if no isosceles is formed."""
    S = []
    for p in order:
        if not _is_isosceles(p, S):
            S.append(p)
    return S

def _make_maximal(S, all_points):
    """Given a no-isosceles set S, make it maximal by greedily adding points."""
    S = list(S)
    S_set = set(S)
    changed = True
    while changed:
        changed = False
        for p in all_points:
            if p in S_set:
                continue
            if not _is_isosceles(p, S):
                S.append(p)
                S_set.add(p)
                changed = True
    return S

def _remove_if_still_maximal(S, all_points):
    """Try to remove points one by one; keep removal if set remains maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if _is_maximal(candidate, all_points):
                S = candidate
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_S = None
    best_size = float('inf')
    
    # Multi-start greedy with different orderings
    orderings = []
    # Row-major
    orderings.append(list(all_points))
    # Column-major
    orderings.append([(x, y) for y in range(N) for x in range(N)])
    # Diagonal order
    diag = []
    for d in range(2 * N - 1):
        for x in range(max(0, d - N + 1), min(N, d + 1)):
            y = d - x
            diag.append((x, y))
    orderings.append(diag)
    # Random orderings
    rng = random.Random(42)
    for _ in range(20):
        o = list(all_points)
        rng.shuffle(o)
        orderings.append(o)
    
    for order in orderings:
        if time.time() > DEADLINE - 60:
            break
        S = _greedy_construct(order, all_points)
        S = _make_maximal(S, all_points)
        if len(S) < best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Local search: try removing points
    if best_S is not None:
        S = _remove_if_still_maximal(best_S, all_points)
        if len(S) < best_size:
            best_size = len(S)
            best_S = list(S)
    
    # More local search iterations if time permits
    if best_S is not None:
        S = list(best_S)
        for _ in range(50):
            if time.time() > DEADLINE - 30:
                break
            S2 = _remove_if_still_maximal(S, all_points)
            if len(S2) < len(S):
                S = S2
            else:
                break
    
    return S if best_S is not None else []
# EVOLVE-BLOCK-END
