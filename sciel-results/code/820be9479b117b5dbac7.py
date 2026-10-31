"""Reference-tour-guided TSP constructive heuristic.

Strategy:
  1. On new instance detection, build a reference tour via multi-start NN + 2-opt.
  2. During construction, guide node selection by following the reference tour
     while incorporating a 2-step lookahead for robustness.
  3. Score per candidate c:
     score(c) = d(cur, c) + beta * ref_penalty(c) + gamma * min_{u in rem\{c}} d(c, u)
     where ref_penalty(c) = 0 if c is the reference tour's next node from cur,
     else d(cur, c) + d(c, ref_next) - d(cur, ref_next) (detour cost).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_last_dm_checksum = None
_ref_tour = None  # list of nodes in reference tour order
_ref_next_map = None  # dict: node -> next node in ref tour
_instance_fitted = False


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start_node):
    """Build a nearest-neighbor tour starting from start_node."""
    n = distance_matrix.shape[0]
    visited = [start_node]
    remaining = set(range(n))
    remaining.discard(start_node)
    current = start_node
    total = 0.0
    
    while remaining:
        rem_list = list(remaining)
        dists = distance_matrix[current, rem_list]
        best_idx = int(np.argmin(dists))
        next_node = rem_list[best_idx]
        total += distance_matrix[current, next_node]
        visited.append(next_node)
        remaining.discard(next_node)
        current = next_node
    
    total += distance_matrix[current, start_node]
    return visited, total


def _two_opt(tour, distance_matrix, max_iter=200):
    """Improve tour using 2-opt local search. Returns improved tour and length."""
    n = len(tour)
    if n < 4:
        return tour, _tour_length(tour, distance_matrix)
    
    tour = list(tour)
    current_len = _tour_length(tour, distance_matrix)
    
    for _ in range(max_iter):
        if _is_deadline_passed():
            break
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip adjacent edges
                if j == i + 1:
                    continue
                # Compute delta for reversing segment i..j
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (distance_matrix[a, c] + distance_matrix[b, d]) - \
                        (distance_matrix[a, b] + distance_matrix[c, d])
                if delta < -1e-10:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    current_len += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return tour, current_len


def _tour_length(tour, distance_matrix):
    """Compute total tour length."""
    n = len(tour)
    if n <= 1:
        return 0.0
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total


def _build_reference_tour(distance_matrix, time_budget=5.0):
    """Build a reference tour using multi-start NN + 2-opt."""
    global _ref_tour, _ref_next_map, _instance_fitted
    
    n = distance_matrix.shape[0]
    if n <= 2:
        _ref_tour = list(range(n))
        _ref_next_map = {i: (i + 1) % n for i in range(n)}
        _instance_fitted = True
        return
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 10.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    # Multi-start: try several starting nodes
    n_starts = min(5, n)
    starts = [0] + np.linspace(1, n - 1, n_starts - 1, dtype=int).tolist()
    
    best_tour = None
    best_len = float('inf')
    
    for start in starts:
        if time.time() > deadline:
            break
        tour, length = _nn_tour(distance_matrix, start)
        if length < best_len:
            best_len = length
            best_tour = tour
        # Apply 2-opt
        tour_opt, len_opt = _two_opt(tour, distance_matrix)
        if len_opt < best_len:
            best_len = len_opt
            best_tour = tour_opt
    
    if best_tour is None:
        best_tour = list(range(n))
    
    _ref_tour = best_tour
    # Build next-node map
    _ref_next_map = {}
    for i in range(n):
        _ref_next_map[best_tour[i]] = best_tour[(i + 1) % n]
    _instance_fitted = True


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick check: is this the same matrix as before?
        step = max(1, n_total // 5)
        checksum = np.sum(distance_matrix[::step, ::step])
        
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True
    
    return False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses reference-tour guidance: follows the reference tour when possible,
    with a 2-step lookahead term for robustness.
    """
    global _instance_fitted, _ref_tour, _ref_next_map
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_fitted = False
        _ref_tour = None
        _ref_next_map = None
        _build_reference_tour(distance_matrix, time_budget=5.0)
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    # Reference tour guidance
    if _ref_next_map is not None and current_node in _ref_next_map:
        ref_next = _ref_next_map[current_node]
        if ref_next in set(unvisited_nodes):
            # Reward following the reference tour
            ref_dist = distance_matrix[current_node, ref_next]
            # Penalty for not following ref: detour cost
            # For candidate c: detour = d(cur,c) + d(c,ref_next) - d(cur,ref_next)
            # We add a penalty proportional to how much worse c is vs ref_next
            beta = 0.3
            for i, node in enumerate(unvisited_nodes):
                if node == ref_next:
                    scores[i] -= beta * ref_dist  # strong reward for ref next
                else:
                    # Detour penalty
                    detour = dists_from_current[i] + distance_matrix[node, ref_next] - ref_dist
                    scores[i] += beta * max(0.0, detour)
    
    # 2-step lookahead: for each candidate, add min distance to other unvisited
    if n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += 0.15 * min_next
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
