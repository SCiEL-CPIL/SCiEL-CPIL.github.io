"""Rewritten program for minimum maximal no-isosceles set on [0,16)^2."""

import itertools
import time
import random
import numpy as np

N = 16
ALL_POINTS = [(i, j) for i in range(N) for j in range(N)]
ALL_POINTS_SET = set(ALL_POINTS)


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(p, chosen_list):
    """Check if p forms an isosceles triangle with any pair in chosen_list."""
    n = len(chosen_list)
    for i in range(n):
        pi = chosen_list[i]
        d_pi = _d2(p, pi)
        for j in range(i + 1, n):
            pj = chosen_list[j]
            if d_pi == _d2(p, pj) or d_pi == _d2(pi, pj) or _d2(p, pj) == _d2(pi, pj):
                return True
    return False


def _can_add(p, chosen_set, chosen_list):
    """Check if p can be added to chosen without forming isosceles triangles."""
    if p in chosen_set:
        return True
    n = len(chosen_list)
    for i in range(n):
        pi = chosen_list[i]
        d_pi = _d2(p, pi)
        for j in range(i + 1, n):
            pj = chosen_list[j]
            if d_pi == _d2(p, pj) or d_pi == _d2(pi, pj) or _d2(p, pj) == _d2(pi, pj):
                return False
    return True


def _is_maximal(chosen_set, chosen_list):
    """Check if no other point can be added."""
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        n = len(chosen_list)
        for i in range(n):
            pi = chosen_list[i]
            d_pi = _d2(p, pi)
            for j in range(i + 1, n):
                pj = chosen_list[j]
                if d_pi == _d2(p, pj) or d_pi == _d2(pi, pj) or _d2(p, pj) == _d2(pi, pj):
                    break
            else:
                # p can be added, so not maximal
                return False
    return True


def _greedy_random_order(seed):
    """Greedy construction with a random permutation of points."""
    rng = random.Random(seed)
    order = list(ALL_POINTS)
    rng.shuffle(order)
    
    chosen = []
    chosen_set = set()
    
    for p in order:
        if _can_add(p, chosen_set, chosen):
            chosen.append(p)
            chosen_set.add(p)
    
    return chosen


def _local_search(chosen, deadline, rng):
    """Try to improve by removing points that allow smaller maximal sets."""
    chosen_set = set(chosen)
    best = list(chosen)
    best_set = set(best)
    best_len = len(best)
    
    while time.time() < deadline:
        # Try removing a random point and see if the set is still maximal
        if len(best) <= 1:
            break
        
        idx = rng.randint(0, len(best) - 1)
        candidate = [x for i, x in enumerate(best) if i != idx]
        candidate_set = set(candidate)
        
        # Check if candidate is still a valid no-isosceles set (it is, since we removed a point)
        # Check if it's maximal
        if _is_maximal(candidate_set, candidate):
            if len(candidate) < best_len:
                best = candidate
                best_set = candidate_set
                best_len = len(best)
        else:
            # Not maximal, so we need to add points to make it maximal
            # Try adding points in random order to make it maximal with minimum additions
            added = list(candidate)
            added_set = set(added)
            
            # Greedily add points to make it maximal
            remaining = [p for p in ALL_POINTS if p not in added_set]
            rng.shuffle(remaining)
            
            for p in remaining:
                if _can_add(p, added_set, added):
                    added.append(p)
                    added_set.add(p)
            
            if len(added) < best_len:
                best = added
                best_set = added_set
                best_len = len(best)
    
    return best


def _simulated_annealing(chosen, deadline, rng):
    """SA-based improvement: try swaps and removals."""
    current = list(chosen)
    current_set = set(current)
    best = list(current)
    best_set = set(current)
    best_len = len(current)
    
    temp = 5.0
    cooling = 0.99
    
    while time.time() < deadline and temp > 0.1:
        # Generate a neighbor: remove a point, then greedily fill to make maximal
        if len(current) > 1:
            idx = rng.randint(0, len(current) - 1)
            candidate = [x for i, x in enumerate(current) if i != idx]
            candidate_set = set(candidate)
            
            # Fill to make maximal
            remaining = [p for p in ALL_POINTS if p not in candidate_set]
            rng.shuffle(remaining)
            
            for p in remaining:
                if _can_add(p, candidate_set, candidate):
                    candidate.append(p)
                    candidate_set.add(p)
            
            new_len = len(candidate)
            old_len = len(current)
            
            if new_len <= old_len or rng.random() < np.exp(-(new_len - old_len) / temp):
                current = candidate
                current_set = candidate_set
                if new_len < best_len:
                    best = list(candidate)
                    best_set = set(candidate)
                    best_len = new_len
        
        temp *= cooling
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts
    best = None
    best_len = float('inf')
    
    while time.time() < deadline - 100:
        for seed in range(100):
            if time.time() > deadline - 100:
                break
            result = _greedy_random_order(seed)
            if len(result) < best_len:
                best = result
                best_len = len(result)
    
    if best is None:
        best = _greedy_random_order(0)
        best_len = len(best)
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 30:
        improved = _local_search(best, deadline - 30, rng)
        if len(improved) < best_len:
            best = improved
            best_len = len(best)
    
    # Phase 3: More randomized restarts with local search
    while time.time() < deadline - 10:
        result = _greedy_random_order(rng.randint(0, 1000000))
        if len(result) < best_len:
            best = result
            best_len = len(result)
        
        if time.time() < deadline - 10:
            improved = _local_search(best, deadline - 10, rng)
            if len(improved) < best_len:
                best = improved
                best_len = len(best)
    
    return best
