"""Improved program: randomized greedy with local search to minimize maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
from math import sqrt


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_iso(p, chosen[i], chosen[j]):
                return False
    return True


def _greedy_random(points, chosen_init=None):
    """Greedy construction in random order. If chosen_init is given, start from it."""
    if chosen_init is not None:
        chosen = list(chosen_init)
        remaining = [p for p in points if p not in set(chosen)]
    else:
        chosen = []
        remaining = list(points)
    
    random.shuffle(remaining)
    
    for p in remaining:
        if _can_add(p, chosen):
            chosen.append(p)
    
    return chosen


def _try_remove(chosen):
    """Try to remove a point from chosen while maintaining no-isosceles property.
    If removal is valid, return the new set (smaller). Otherwise return None."""
    n = len(chosen)
    for i in range(n):
        candidate = chosen[i]
        new_set = [chosen[j] for j in range(n) if j != i]
        # Check if new_set is still no-isosceles
        valid = True
        for a_idx in range(len(new_set)):
            for b_idx in range(a_idx + 1, len(new_set)):
                for c_idx in range(b_idx + 1, len(new_set)):
                    if _is_iso(new_set[a_idx], new_set[b_idx], new_set[c_idx]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return new_set
    return None


def _local_search(chosen, max_iters=50):
    """Repeatedly try to remove points to minimize the set while keeping it no-isosceles."""
    for _ in range(max_iters):
        new_set = _try_remove(chosen)
        if new_set is None:
            break
        chosen = new_set
    return chosen


def _is_maximal(chosen, all_points):
    """Check if chosen is maximal: no further point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _make_maximal(chosen, all_points):
    """Add points to make the set maximal (greedy, random order)."""
    chosen = list(chosen)
    remaining = [p for p in all_points if p not in set(chosen)]
    random.shuffle(remaining)
    for p in remaining:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_size = len(all_points) + 1
    best_set = None
    
    # First, do a few random greedy restarts
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        chosen = _greedy_random(all_points)
        # Make sure it's maximal (greedy should produce maximal, but verify)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen
    
    if best_set is None:
        # Fallback: fixed order greedy
        chosen = []
        for p in all_points:
            if _can_add(p, chosen):
                chosen.append(p)
        return chosen
    
    # Now do local search on the best set found
    # Try to remove points from the best set
    improved = _local_search(best_set, max_iters=100)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved
    
    # Do more restarts with local search
    for r in range(500):
        if time.time() > deadline:
            break
        chosen = _greedy_random(all_points)
        if len(chosen) >= best_size:
            continue
        # Try local search
        improved = _local_search(chosen, max_iters=50)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    # Final: try to make best_set maximal if it isn't (shouldn't happen from greedy, but safe)
    if not _is_maximal(best_set, all_points):
        best_set = _make_maximal(best_set, all_points)
    
    return best_set
# EVOLVE-BLOCK-END
