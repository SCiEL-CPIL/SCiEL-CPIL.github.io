"""Rewritten: CSP-style local search to minimize maximal no-isosceles set."""
import random
import time
import numpy as np

N = 16
DEADLINE = 320.0  # safety margin before 330s kill

# Precompute all points
_ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
_POINT_SET = set(_ALL_POINTS)
_POINT_TO_IDX = {p: i for i, p in enumerate(_ALL_POINTS)}
_NUM_POINTS = N * N  # 256


def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _sq_dist(a, b)
    d2 = _sq_dist(a, c)
    d3 = _sq_dist(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _greedy_construct(rng, order=None):
    """Randomized greedy: build a maximal no-isosceles set."""
    if order is None:
        order = list(_ALL_POINTS)
        rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p in order:
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _is_isosceles(p, chosen[i], chosen[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set


def _is_maximal(chosen, chosen_set, rng=None, time_limit=1.0):
    """Check if chosen is a maximal no-isosceles set.
    For every p not in chosen_set, there must exist a,b in chosen such that
    {p,a,b} is isosceles."""
    candidates = [p for p in _ALL_POINTS if p not in chosen_set]
    for p in candidates:
        covered = False
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _is_isosceles(p, chosen[i], chosen[j]):
                    covered = True
                    break
            if covered:
                break
        if not covered:
            return False, p
    return True, None


def _local_search_minimize(chosen, chosen_set, rng, deadline):
    """Try to remove points while maintaining maximality.
    Returns (chosen, chosen_set, removed_count)."""
    removed = 0
    while time.time() < deadline:
        # Try each point for removal
        progress = False
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() >= deadline:
                break
            p = chosen[idx]
            # Try removing p
            new_set = chosen_set - {p}
            new_list = [c for i, c in enumerate(chosen) if i != idx]
            # Check maximality of new_list
            is_max, _ = _is_maximal(new_list, new_set, rng, time_limit=0.5)
            if is_max:
                chosen = new_list
                chosen_set = new_set
                removed += 1
                progress = True
                break  # restart the shuffle after a removal
        if not progress:
            break
    return chosen, chosen_set, removed


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best = None
    best_size = _NUM_POINTS + 1
    
    # Phase 1: Multiple randomized greedy restarts
    phase1_deadline = start_time + 60.0
    attempt = 0
    while time.time() < phase1_deadline:
        attempt += 1
        chosen, chosen_set = _greedy_construct(rng)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = (list(chosen), set(chosen_set))
    
    # Phase 2: Local search on the best solution found
    if best is not None:
        phase2_deadline = min(deadline - 5.0, start_time + 200.0)
        chosen, chosen_set, _ = _local_search_minimize(
            best[0], best[1], rng, phase2_deadline
        )
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = (list(chosen), set(chosen_set))
    
    # Phase 3: More restarts with local search if time remains
    while time.time() < deadline - 10.0:
        chosen, chosen_set = _greedy_construct(rng)
        if len(chosen) >= best_size:
            continue
        # Local search on this new solution
        local_deadline = min(deadline - 5.0, time.time() + 30.0)
        chosen, chosen_set, _ = _local_search_minimize(
            chosen, chosen_set, rng, local_deadline
        )
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = (list(chosen), set(chosen_set))
    
    if best is None:
        # Fallback: deterministic greedy
        chosen, chosen_set = _greedy_construct(rng, order=list(_ALL_POINTS))
        return chosen
    
    return best[0]
