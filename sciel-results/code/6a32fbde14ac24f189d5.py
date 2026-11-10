"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute():
    """Precompute all points and distance matrix."""
    all_points = list(itertools.product(range(N), repeat=2))
    n_pts = len(all_points)
    dist = [[0] * n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        xi, yi = all_points[i]
        for j in range(i + 1, n_pts):
            xj, yj = all_points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist[i][j] = d
            dist[j][i] = d
    return all_points, dist, n_pts


def _is_isos(p, i, j, dist):
    """Check if points p, i, j form an isosceles triangle."""
    d1 = dist[p][i]
    d2 = dist[p][j]
    d3 = dist[i][j]
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen, dist):
    """Check if point p can be added to chosen without creating isosceles."""
    for idx_i in range(len(chosen)):
        for idx_j in range(idx_i + 1, len(chosen)):
            if _is_isos(p, chosen[idx_i], chosen[idx_j], dist):
                return False
    return True


def _is_maximal(chosen, all_points, dist, n_pts):
    """Check if chosen is a maximal no-isosceles set."""
    chosen_set = set(chosen)
    for p in range(n_pts):
        if p in chosen_set:
            continue
        if _can_add(p, chosen, dist):
            return False
    return True


def _greedy_random(all_points, dist, n_pts, rng):
    """Greedy construction with random ordering."""
    order = list(range(n_pts))
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen, dist):
            chosen.append(p)
    return chosen


def _local_search_minimize(chosen, dist, n_pts, max_iter=50):
    """Try to reduce the set size while maintaining maximality."""
    for _ in range(max_iter):
        improved = False
        for idx in range(len(chosen) - 1, -1, -1):
            p = chosen[idx]
            trial = chosen[:idx] + chosen[idx + 1:]
            if _is_maximal(trial, None, dist, n_pts):
                chosen = trial
                improved = True
                break
        if not improved:
            break
    return chosen


def construct():
    all_points, dist, n_pts = _precompute()
    
    best_set = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 280
    
    # Run many randomized trials with local search
    for trial in range(500):
        if time.time() - start_time > time_limit:
            break
        rng = random.Random(trial * 7919 + 31)
        
        # Random greedy
        chosen = _greedy_random(all_points, dist, n_pts, rng)
        
        # Local search to minimize
        chosen = _local_search_minimize(chosen, dist, n_pts)
        
        # Verify maximality
        if not _is_maximal(chosen, all_points, dist, n_pts):
            continue
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    # Fallback: deterministic greedy
    if best_set is None:
        chosen = []
        for p in range(n_pts):
            if _can_add(p, chosen, dist):
                chosen.append(p)
        best_set = chosen
    
    return [all_points[i] for i in best_set]
# EVOLVE-BLOCK-END
