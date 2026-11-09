"""Rewritten program for minimum maximal no-isosceles set task."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
_TOTAL = N * N
_DEADLINE = time.time() + 280.0

# Precompute all points
_ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _build_conflict_pairs():
    """
    For each point p_idx, return a frozenset of frozenset({i, j}) such that
    (p_idx, i, j) forms an isosceles triangle.
    We store as a list of sets of (min, max) tuples for fast lookup.
    """
    conflicts = [set() for _ in range(_TOTAL)]
    points = _ALL_POINTS
    for p_idx in range(_TOTAL):
        p = points[p_idx]
        for i in range(_TOTAL):
            if i == p_idx:
                continue
            a = points[i]
            for j in range(i + 1, _TOTAL):
                if j == p_idx:
                    continue
                b = points[j]
                d1 = _d2(p, a)
                d2 = _d2(p, b)
                d3 = _d2(a, b)
                if d1 == d2 or d1 == d3 or d2 == d3:
                    conflicts[p_idx].add((i, j))
    return conflicts


# Precompute once at module level to avoid recomputation
_CONFLICTS = _build_conflict_pairs()


def _can_add(p_idx, chosen_set):
    """Check if p_idx can be added to chosen_set without creating an isosceles triangle."""
    for (i, j) in _CONFLICTS[p_idx]:
        if i in chosen_set and j in chosen_set:
            return False
    return True


def _is_maximal(chosen_set):
    """Check that no additional grid point can be added."""
    for p_idx in range(_TOTAL):
        if p_idx in chosen_set:
            continue
        if _can_add(p_idx, chosen_set):
            return False
    return True


def _is_valid_no_isosceles(chosen_set):
    """Verify no three points in chosen_set form an isosceles triangle."""
    for p_idx in chosen_set:
        for (i, j) in _CONFLICTS[p_idx]:
            if i in chosen_set and j in chosen_set:
                return False
    return True


def _greedy_row_major():
    """Greedy construction in row-major order."""
    chosen = []
    chosen_set = set()
    for p_idx in range(_TOTAL):
        if _can_add(p_idx, chosen_set):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return chosen, chosen_set


def _greedy_with_order(order):
    """Greedy construction with a given point order."""
    chosen = []
    chosen_set = set()
    for p_idx in order:
        if _can_add(p_idx, chosen_set):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return chosen, chosen_set


def _local_search_reduce(chosen_list, chosen_set):
    """
    Try to remove points from the set while maintaining no-isosceles and maximality.
    Greedily remove a point if the set remains maximal without it.
    """
    current = list(chosen_list)
    current_set = set(chosen_list)
    
    while True:
        if time.time() > _DEADLINE:
            break
        removed = False
        for p_idx in list(current):
            new_set = current_set - {p_idx}
            # Check maximality: no point can be added
            is_still_maximal = True
            for q_idx in range(_TOTAL):
                if q_idx in new_set:
                    continue
                if _can_add(q_idx, new_set):
                    is_still_maximal = False
                    break
            if is_still_maximal:
                current.remove(p_idx)
                current_set = new_set
                removed = True
                break
        if not removed:
            break
    
    return current, current_set


def _random_greedy(seed):
    """Random order greedy."""
    rng = random.Random(seed)
    order = list(range(_TOTAL))
    rng.shuffle(order)
    return _greedy_with_order(order)


def _multi_start_greedy(n_starts=30):
    """Try multiple random greedy starts and keep the best."""
    best_size = _TOTAL + 1
    best_list = None
    best_set = None
    
    for seed in range(n_starts):
        if time.time() > _DEADLINE - 10:
            break
        lst, st = _random_greedy(seed)
        if len(lst) < best_size:
            best_size = len(lst)
            best_list = lst
            best_set = st
    
    # Also try row-major
    lst, st = _greedy_row_major()
    if len(lst) < best_size:
        best_size = len(lst)
        best_list = lst
        best_set = st
    
    return best_list, best_set


def _pairwise_swap_search(chosen_list, chosen_set):
    """
    Try swapping one point for another to reduce size.
    For each point in the set, try removing it and adding a different point,
    checking if the result is still valid and maximal with smaller size.
    """
    current = list(chosen_list)
    current_set = set(chosen_list)
    improved = True
    
    while improved and time.time() < _DEADLINE - 20:
        improved = False
        for p_idx in list(current):
            # Try removing p_idx
            new_set = current_set - {p_idx}
            # Try adding each point not in new_set
            for q_idx in range(_TOTAL):
                if q_idx in new_set:
                    continue
                if not _can_add(q_idx, new_set):
                    continue
                candidate_set = new_set | {q_idx}
                # Check no-isosceles
                valid = True
                for x_idx in candidate_set:
                    for (i, j) in _CONFLICTS[x_idx]:
                        if i in candidate_set and j in candidate_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check maximality
                if not _is_maximal(candidate_set):
                    continue
                # If same size, we could swap but it doesn't reduce size
                # We only care about reducing size, so skip
            # Try removing p_idx only (if still maximal)
            if _is_maximal(new_set):
                current.remove(p_idx)
                current_set = new_set
                improved = True
    
    return current, current_set


def construct():
    """
    Construct a minimal maximal no-isosceles set.
    Returns a list of (x, y) grid points.
    """
    if time.time() > _DEADLINE:
        return [_ALL_POINTS[0]]
    
    # Multi-start greedy
    best_list, best_set = _multi_start_greedy(n_starts=50)
    
    # Local search to reduce
    if time.time() < _DEADLINE - 30:
        best_list, best_set = _local_search_reduce(best_list, best_set)
    
    # Pairwise improvement
    if time.time() < _DEADLINE - 15:
        best_list, best_set = _pairwise_swap_search(best_list, best_set)
    
    # Final local search pass
    if time.time() < _DEADLINE - 5:
        best_list, best_set = _local_search_reduce(best_list, best_set)
    
    return [_ALL_POINTS[i] for i in best_list]
