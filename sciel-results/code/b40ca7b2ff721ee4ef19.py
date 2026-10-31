"""Multi-Start NN + 2-Opt guided TSP constructive heuristic.

Strategy:
  1. For each new instance, pre-compute a high-quality reference tour using
     Multi-Start Nearest-Neighbor (all 50 starts) followed by 2-Opt to convergence.
  2. During the constructive phase, select the next node by:
     - Strong pull towards the successor in the reference tour.
     - Modest pull towards the destination.
     - A connectivity term to avoid isolating far nodes.
  3. Vectorized scoring for speed.

This is a structural change from parameter-fitting heuristics to a
reference-tour-following approach, inspired by the multi-start + 2-opt finding.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 145.0  # Safety margin before 150s limit

# Per-instance state
_instance_counter = 0
_reference_tour = None  # list of nodes in reference tour order (cyclic)
_successor_map = None   # dict: node -> successor in reference tour
_instance_key = None


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start):
    """Nearest-neighbor tour starting from 'start'."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        d = distance_matrix[current].copy()
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        tour.append(nxt)
        visited[nxt] = True
        current = nxt
    return tour


def _two_opt(tour, distance_matrix):
    """Improve tour using 2-opt to convergence."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    improved = True
    # Precompute distance matrix reference
    dm = distance_matrix
    while improved:
        improved = False
        for i in range(1, n - 1):
            if _is_deadline_passed():
                break
            for j in range(i + 1, n):
                if _is_deadline_passed():
                    break
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                if a == c or a == d or b == c or b == d:
                    continue
                delta = (dm[a][c] + dm[b][d]) - (dm[a][b] + dm[c][d])
                if delta < -1e-10:
                    # Reverse segment i..j
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if _is_deadline_passed():
            break
    return tour


def _build_reference_tour(distance_matrix):
    """Build a high-quality reference tour using multi-start NN + 2-Opt."""
    n = distance_matrix.shape[0]
    
    best_tour = None
    best_len = float('inf')
    
    # Multi-start NN: try all starting nodes
    for start in range(n):
        if _is_deadline_passed():
            break
        tour = _nn_tour(distance_matrix, start)
        # Compute length
        length = sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
        if length < best_len:
            best_len = length
            best_tour = tour
    
    if best_tour is None:
        # Fallback: simple NN from 0
        best_tour = _nn_tour(distance_matrix, 0)
        best_len = sum(distance_matrix[best_tour[i]][best_tour[(i + 1) % n]] for i in range(n))
    
    # Apply 2-Opt to the best NN tour
    improved_tour = _two_opt(best_tour, distance_matrix)
    
    # Verify length
    final_len = sum(distance_matrix[improved_tour[i]][improved_tour[(i + 1) % n]] for i in range(n))
    
    return improved_tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses a reference tour (built via Multi-Start NN + 2-Opt) to guide
    the constructive heuristic. Strongly prefers the successor in the
    reference tour, with secondary terms for destination proximity and
    connectivity.
    """
    global _instance_counter, _reference_tour, _successor_map, _instance_key
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)
    
    # Detect new instance
    if is_new_instance:
        # Check if we already have a reference tour for this instance
        # Use a simple key: the first few entries of the distance matrix
        key = (distance_matrix[0, 1], distance_matrix[1, 2], distance_matrix[2, 3] if n_total > 3 else 0.0)
        if _instance_key != key or _reference_tour is None:
            _instance_counter += 1
            _instance_key = key
            _reference_tour = _build_reference_tour(distance_matrix)
            # Build successor map
            _successor_map = {}
            n_ref = len(_reference_tour)
            for i in range(n_ref):
                _successor_map[_reference_tour[i]] = _reference_tour[(i + 1) % n_ref]
    
    if _is_deadline_passed():
        # Fallback: nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    if _reference_tour is None or _successor_map is None:
        # Fallback: nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Vectorized scoring
    unvisited_arr = np.asarray(unvisited_nodes)
    dists_from_current = distance_matrix[current_node, unvisited_arr]
    
    # Primary: distance to current (NN component)
    scores = dists_from_current.copy()
    
    # Strong pull towards reference tour successor
    if current_node in _successor_map:
        ref_successor = _successor_map[current_node]
        # Find if ref_successor is in unvisited
        try:
            ref_idx = np.where(unvisited_arr == ref_successor)[0][0]
            # Heavily discount the score for the reference successor
            scores[ref_idx] *= 0.1  # 90% discount
        except (IndexError, ValueError):
            pass
    
    # Secondary: distance to destination (modest weight)
    dists_to_dest = distance_matrix[unvisited_arr, destination_node]
    scores += 0.15 * dists_to_dest
    
    # Tertiary: connectivity - prefer nodes that are close to other unvisited nodes
    # (avoids isolating far nodes)
    if n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += 0.05 * min_next
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
