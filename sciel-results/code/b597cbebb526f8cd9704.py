"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random

_DEADLINE = time.time() + 280

_all_points = list(itertools.product(range(N), repeat=2))
_NUM_POINTS = N * N

# Precompute squared distances
_dist2 = [[0] * _NUM_POINTS for _ in range(_NUM_POINTS)]
for i in range(_NUM_POINTS):
    for j in range(i + 1, _NUM_POINTS):
        d = ( _all_points[i][0] - _all_points[j][0] ) ** 2 + ( _all_points[i][1] - _all_points[j][1] ) ** 2
        _dist2[i][j] = d
        _dist2[j][i] = d

# For each point p, precompute the set of pairs (i,j) with i<j such that
# (p, all_points[i], all_points[j]) is isosceles.
# Store as a set of frozensets or as a set of (min, max) tuples.
_conflicts = [set() for _ in range(_NUM_POINTS)]
for p in range(_NUM_POINTS):
    for i in range(_NUM_POINTS):
        if i == p:
            continue
        for j in range(i + 1, _NUM_POINTS):
            if j == p:
                continue
            d_pi = _dist2[p][i]
            d_pj = _dist2[p][j]
            d_ij = _dist2[i][j]
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                _conflicts[p].add((min(i, j), max(i, j)))

def _can_add_idx(p_idx, chosen_indices):
    """Check if point p_idx can be added to chosen_indices without creating isosceles."""
    # For each pair in chosen, check if (p, a, b) is isosceles
    for i in range(len(chosen_indices)):
        for j in range(i + 1, len(chosen_indices)):
            a = chosen_indices[i]
            b = chosen_indices[j]
            d_pa = _dist2[p_idx][a]
            d_pb = _dist2[p_idx][b]
            d_ab = _dist2[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def _is_blocked_idx(p_idx, chosen_indices):
    """Check if p_idx is blocked by chosen_indices."""
    if p_idx in chosen_indices:
        return True
    for i in range(len(chosen_indices)):
        for j in range(i + 1, len(chosen_indices)):
            a = chosen_indices[i]
            b = chosen_indices[j]
            d_pa = _dist2[p_idx][a]
            d_pb = _dist2[p_idx][b]
            d_ab = _dist2[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _is_maximal(chosen_indices):
    """Check if chosen_indices is maximal."""
    chosen_set = set(chosen_indices)
    for p in range(_NUM_POINTS):
        if p not in chosen_set:
            if _can_add_idx(p, chosen_indices):
                return False
    return True

def _construct_greedy(order_indices):
    """Greedy construction given an order of indices."""
    chosen = []
    for p_idx in order_indices:
        if _can_add_idx(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _minimize_maximal(chosen_indices):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen_indices)
    improved = True
    while improved:
        improved = False
        for i in range(len(chosen) - 1, -1, -1):
            candidate = chosen[:i] + chosen[i+1:]
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen

def construct():
    start_time = time.time()
    
    best = None
    best_size = _NUM_POINTS + 1
    
    rng = random.Random(42)
    
    while time.time() - start_time < _DEADLINE - 30:
        # Generate a random permutation
        order = list(range(_NUM_POINTS))
        rng.shuffle(order)
        
        # Greedy construction
        chosen = _construct_greedy(order)
        
        # Minimize
        minimized = _minimize_maximal(chosen)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best = minimized
        
        if best_size <= 10:
            break
    
    if best is None:
        best = _construct_greedy(list(range(_NUM_POINTS)))
        best = _minimize_maximal(best)
    
    return [_all_points[i] for i in best]
# EVOLVE-BLOCK-END
