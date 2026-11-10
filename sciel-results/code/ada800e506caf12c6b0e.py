"""
Minimum maximal no-isosceles set on N x N grid.
Combines vectorized feasibility checks (inspiration 2) with efficient greedy
construction, pruning, and local search (current program + inspiration 1).
"""
import time
import random
import itertools
import numpy as np

N = 16
DEADLINE = time.time() + 280  # safety margin before 330s kill


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _can_add(p, chosen, dist_cache):
    """
    Check if p can be added to chosen without forming an isosceles triangle.
    Uses dist_cache: dict mapping (i,j) -> dist2 for pairs in chosen.
    """
    k = len(chosen)
    if k < 2:
        return True
    
    # Compute distances from p to each point in chosen
    dp = [_dist2(p, a) for a in chosen]
    
    # Check: d(p, a_i) == d(p, a_j) for any i != j
    seen = set()
    for d in dp:
        if d in seen:
            return False
        seen.add(d)
    
    # Check: d(p, a_i) == d(a_i, a_j) or d(p, a_j) == d(a_i, a_j)
    for i in range(k):
        for j in range(i + 1, k):
            d_ij = dist_cache.get((i, j))
            if d_ij is None:
                d_ij = _dist2(chosen[i], chosen[j])
                dist_cache[(i, j)] = d_ij
            if dp[i] == d_ij or dp[j] == d_ij:
                return False
    return True


def _can_add_vectorized(p, chosen_arr, d_pair_triu):
    """
    Vectorized check: can p be added to chosen (as numpy array) without isosceles?
    chosen_arr: (m, 2) int array
    d_pair_triu: array of pairwise squared distances for i<j pairs
    """
    m = len(chosen_arr)
    if m < 2:
        return True
    
    p_arr = np.asarray(p, dtype=np.int64)
    diff = chosen_arr - p_arr
    d_new = (diff ** 2).sum(axis=1)  # (m,)
    
    # Check d_new[i] == d_new[j]
    if len(set(d_new.tolist())) != m:
        return False
    
    # Check d_new[i] == d_pair[i,j] or d_new[j] == d_pair[i,j]
    # d_pair_triu is indexed by (i, j) with i < j
    iu = np.triu_indices(m, k=1)
    dn_i = d_new[iu[0]]
    dn_j = d_new[iu[1]]
    bad = (dn_i == d_pair_triu) | (dn_j == d_pair_triu)
    return not np.any(bad)


def _build_pair_dist(chosen):
    """Build dictionary of pairwise squared distances for chosen list."""
    k = len(chosen)
    cache = {}
    for i in range(k):
        for j in range(i + 1, k):
            cache[(i, j)] = _dist2(chosen[i], chosen[j])
    return cache


def _greedy_maximal(order, deadline):
    """Greedy maximal construction given an ordering of points."""
    chosen = []
    dist_cache = {}
    for p in order:
        if time.time() > deadline:
            return None
        if _can_add(p, chosen, dist_cache):
            chosen.append(p)
            # Add new distances to cache
            idx = len(chosen) - 1
            for j in range(idx):
                dist_cache[(j, idx)] = _dist2(chosen[j], chosen[idx])
    return chosen


def _is_maximal(chosen, all_points):
    """Check if no point can be added to chosen without creating an isosceles triangle."""
    chosen_set = set(chosen)
    k = len(chosen)
    if k < 2:
        # Any point can be added
        for p in all_points:
            if p not in chosen_set:
                return False
        return True
    
    # Precompute pairwise distances
    dist_cache = _build_pair_dist(chosen)
    
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen, dist_cache):
            return False
    return True


def _prune(chosen, all_points, deadline):
    """Try to remove points that are not essential for maximality."""
    if time.time() > deadline:
        return chosen
    chosen = list(chosen)
    i = 0
    while i < len(chosen):
        if time.time() > deadline:
            break
        candidate = chosen[:i] + chosen[i+1:]
        if _is_maximal(candidate, all_points):
            chosen = candidate
            # Don't increment i since we removed an element
        else:
            i += 1
    return chosen


def _local_search(chosen, all_points, deadline):
    """
    Local search: try to swap out a point for another to reduce size while
    maintaining maximality. Also try deletions.
    """
    chosen = list(chosen)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(chosen)
        if n == 0:
            break
        
        chosen_set = set(chosen)
        dist_cache = _build_pair_dist(chosen)
        
        # Try deletions
        for i in range(n):
            if time.time() > deadline - 5:
                return chosen
            candidate = chosen[:i] + chosen[i+1:]
            if _is_maximal(candidate, all_points):
                chosen = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i in range(n):
            if time.time() > deadline - 5:
                return chosen
            base = chosen[:i] + chosen[i+1:]
            base_set = set(base)
            for p in all_points:
                if p in base_set:
                    continue
                # Check if p can be added to base
                if _can_add(p, base, {}):
                    candidate = base + [p]
                    if _is_maximal(candidate, all_points):
                        chosen = candidate
                        improved = True
                        break
            if improved:
                break
    
    return chosen


def construct():
    deadline = time.time() + 280
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # First: deterministic row-major greedy
    result = _greedy_maximal(all_points, deadline)
    if result is not None:
        result = _prune(result, all_points, deadline)
        result = _local_search(result, all_points, deadline)
        best = result
        best_size = len(result)
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    num_trials = 3000
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        # Shuffle the order
        order = all_points[:]
        rng.shuffle(order)
        result = _greedy_maximal(order, deadline)
        if result is None:
            break
        # Prune the result
        result = _prune(result, all_points, deadline)
        # Local search
        result = _local_search(result, all_points, deadline)
        if len(result) < best_size:
            best = result
            best_size = len(result)
            # If we've found a very small set, we can stop earlier
            if best_size <= 8:
                break
    
    return best if best is not None else []
