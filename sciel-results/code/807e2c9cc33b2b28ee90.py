"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_DEADLINE = time.time() + 280.0

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _build_initial_maximal(points, order):
    """Build a maximal no-isosceles set by greedily adding points in given order."""
    chosen = []
    chosen_set = set()
    for p in order:
        if p in chosen_set:
            continue
        # Check if adding p creates an isosceles triangle with any pair in chosen
        valid = True
        for i, a in enumerate(chosen):
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                if _is_isosceles(p, a, b):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _check_maximal(chosen, all_points, isos_pairs_by_point):
    """Check if chosen is maximal: every unselected point forms isosceles with some pair in chosen."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        # p must form isosceles with some pair (a,b) both in chosen
        pairs = isos_pairs_by_point.get(p, [])
        found = False
        for (a, b) in pairs:
            if a in chosen_set and b in chosen_set:
                found = True
                break
        if not found:
            return False
    return True

def _precompute_isos_pairs(all_points):
    """For each point p, precompute all pairs (a,b) such that (p,a,b) is isosceles."""
    n = len(all_points)
    isos_pairs = {p: [] for p in all_points}
    for i in range(n):
        p = all_points[i]
        for j in range(n):
            if j == i:
                continue
            a = all_points[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                b = all_points[k]
                if _is_isosceles(p, a, b):
                    isos_pairs[p].append((a, b))
    return isos_pairs

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    
    # Precompute isosceles pairs for each point
    isos_pairs_by_point = _precompute_isos_pairs(all_points)
    
    best_set = None
    best_size = n_points + 1
    
    # Try multiple random orderings
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > _DEADLINE - 30:
            break
        
        order = all_points[:]
        random.shuffle(order)
        
        # Build initial maximal set
        chosen = _build_initial_maximal(all_points, order)
        
        # Now try to minimize: remove points while maintaining maximality
        changed = True
        while changed:
            if time.time() > _DEADLINE - 30:
                break
            changed = False
            for idx in range(len(chosen)):
                if time.time() > _DEADLINE - 30:
                    break
                p = chosen[idx]
                candidate = chosen[:idx] + chosen[idx+1:]
                if _check_maximal(candidate, all_points, isos_pairs_by_point):
                    chosen = candidate
                    changed = True
                    break  # Restart loop since indices changed
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    if best_set is None:
        # Fallback: greedy in row-major order
        best_set = _build_initial_maximal(all_points, all_points)
    
    return best_set
# EVOLVE-BLOCK-END
