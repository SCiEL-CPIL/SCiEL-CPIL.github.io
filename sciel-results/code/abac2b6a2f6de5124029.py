import itertools
import time
import random
import sys
from functools import lru_cache

N = 16
_DEADLINE = time.time() + 280
TOTAL_POINTS = N * N  # 256

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_distances(all_points):
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix

def _can_add_fast(x, chosen_list, dist_matrix, chosen_dist_sets):
    """
    Fast check using precomputed distance sets.
    chosen_dist_sets[i] is the set of distances from chosen_list[i] to other points in chosen_list.
    """
    k = len(chosen_list)
    if k == 0:
        return True
    
    # Distances from x to all chosen must be distinct
    seen_dists = set()
    for p in chosen_list:
        d = dist_matrix[x][p]
        if d in seen_dists:
            return False
        seen_dists.add(d)
    
    # For each p in chosen, d(x,p) must not be in p's existing distance set
    for idx, p in enumerate(chosen_list):
        d_xp = dist_matrix[x][p]
        if d_xp in chosen_dist_sets[idx]:
            return False
            
    return True

def _update_dist_sets_on_add(x, chosen_list, dist_matrix, chosen_dist_sets):
    """
    Update chosen_dist_sets after adding x to chosen_list.
    Returns new list of dist sets.
    """
    k = len(chosen_list)
    new_dist_sets = [set(s) for s in chosen_dist_sets]
    for i in range(k):
        p = chosen_list[i]
        d = dist_matrix[x][p]
        new_dist_sets[i].add(d)
    new_set_for_x = set()
    for p in chosen_list:
        new_set_for_x.add(dist_matrix[x][p])
    new_dist_sets.append(new_set_for_x)
    return new_dist_sets

def _is_maximal_fast(chosen_list, all_indices, dist_matrix, chosen_dist_sets):
    """
    Check if chosen is maximal: no point outside chosen can be added.
    """
    chosen_set = set(chosen_list)
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add_fast(x, chosen_list, dist_matrix, chosen_dist_sets):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, rng):
    """
    Construct a maximal set by adding points in random order.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen_list = []
    chosen_dist_sets = []
    for p in order:
        if _can_add_fast(p, chosen_list, dist_matrix, chosen_dist_sets):
            chosen_list.append(p)
            chosen_dist_sets = _update_dist_sets_on_add(p, chosen_list[:-1], dist_matrix, chosen_dist_sets)
    return chosen_list, chosen_dist_sets

def _remove_and_remaximize(chosen_list, chosen_dist_sets, all_indices, dist_matrix, deadline, rng):
    """
    Try to remove one point, then greedily re-maximize the set.
    Return the new set if it's smaller, else None.
    """
    n = len(chosen_list)
    
    for i in range(n):
        if time.time() > deadline:
            return None
        p = chosen_list[i]
        
        # Construct base set without p
        base_indices = [chosen_list[k] for k in range(n) if k != i]
        m = len(base_indices)
        base_dist_sets = [set() for _ in range(m)]
        for a in range(m):
            for b in range(a+1, m):
                d = dist_matrix[base_indices[a]][base_indices[b]]
                base_dist_sets[a].add(d)
                base_dist_sets[b].add(d)
        
        # Greedily add points to base to make it maximal
        base_set = set(base_indices)
        candidates = [x for x in all_indices if x not in base_set]
        rng.shuffle(candidates)
        
        new_list = list(base_indices)
        new_dist_sets = [set(s) for s in base_dist_sets]
        
        for r in candidates:
            if time.time() > deadline:
                return None
            if _can_add_fast(r, new_list, dist_matrix, new_dist_sets):
                new_list.append(r)
                new_dist_sets = _update_dist_sets_on_add(r, new_list[:-1], dist_matrix, new_dist_sets)
        
        # Check if new_list is maximal
        if _is_maximal_fast(new_list, all_indices, dist_matrix, new_dist_sets):
            if len(new_list) < n:
                return new_list, new_dist_sets
    
    return None

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix = _precompute_distances(all_points)
    
    best_indices = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < _DEADLINE:
        # 1. Greedy Construction
        chosen_indices, chosen_dist_sets = _construct_greedy(all_indices, dist_matrix, rng)
        
        # 2. Try to reduce size by removing points and re-maximizing
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            result = _remove_and_remaximize(chosen_indices, chosen_dist_sets, all_indices, dist_matrix, _DEADLINE, rng)
            if result is not None:
                chosen_indices, chosen_dist_sets = result
                improved = True
        
        # 3. Update best
        if len(chosen_indices) < best_size:
            best_size = len(chosen_indices)
            best_indices = chosen_indices
            
    if best_indices is None:
        best_indices, _ = _construct_greedy(all_indices, dist_matrix, rng)
        
    return [all_points[i] for i in best_indices]
