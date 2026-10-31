"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta) via a small
     grid search using a fast approximate tour builder.
  2. Use a vectorized 2-step lookahead with soft-min (log-sum-exp) for
     smoothness and better gradient-like behavior.
  3. Add a smooth connectivity/isolation penalty term.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*softmin_{u in rem\\{c}} d(c,u)
             + delta * (d(c,dest) - softmin_u d(u,dest))  [isolation penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state: precomputed tour sequence
_instance_tour = None
_instance_id_key = None
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start_node):
    """Build a nearest-neighbor tour from start_node. Returns list of nodes in order."""
    n = distance_matrix.shape[0]
    tour = [start_node]
    unvisited = set(range(n)) - {start_node}
    current = start_node
    while unvisited:
        # Find nearest unvisited
        min_dist = float('inf')
        next_node = -1
        for node in unvisited:
            d = distance_matrix[current, node]
            if d < min_dist:
                min_dist = d
                next_node = node
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour


def _two_opt(tour, distance_matrix, max_passes=50):
    """Apply 2-opt local search to improve the tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    
    best_tour = list(tour)
    best_len = _tour_length(best_tour, distance_matrix)
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if _is_deadline_passed():
                return best_tour
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                # Compute delta for 2-opt move
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-10:
                    # Reverse segment i+1 to j
                    best_tour[i + 1:j + 1] = reversed(best_tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
    return best_tour


def _tour_length(tour, distance_matrix):
    """Compute total length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total


def _greedy_insertion_tour(distance_matrix, start_node):
    """Build a tour using greedy insertion heuristic.
    
    Start with the two closest cities to start_node, then insert each remaining
    city at the position that minimizes the increase in tour length.
    """
    n = distance_matrix.shape[0]
    
    # Find the city closest to start_node
    dists_from_start = distance_matrix[start_node, :]
    dists_from_start[start_node] = float('inf')
    second_city = int(np.argmin(dists_from_start))
    
    # Initial tour: start -> second_city -> start
    tour = [start_node, second_city]
    remaining = set(range(n)) - {start_node, second_city}
    
    while remaining:
        if _is_deadline_passed():
            break
        
        # For each remaining city, find best insertion position
        best_city = -1
        best_pos = -1
        best_increase = float('inf')
        
        rem_list = list(remaining)
        for city in rem_list:
            # Try inserting at each position in the current tour
            tour_len = len(tour)
            for pos in range(tour_len):
                prev_node = tour[pos]
                next_node = tour[(pos + 1) % tour_len]
                increase = distance_matrix[prev_node, city] + distance_matrix[city, next_node] - distance_matrix[prev_node, next_node]
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = pos
        
        if best_city == -1:
            break
        
        # Insert best_city at best_pos
        tour.insert(best_pos + 1, best_city)
        remaining.remove(best_city)
    
    return tour


def _build_reference_tour(distance_matrix, start_node, time_budget=8.0):
    """Build a high-quality reference tour using multiple strategies and 2-opt.
    
    Returns the best tour found.
    """
    n = distance_matrix.shape[0]
    if n <= 2:
        return list(range(n))
    
    deadline = time.time() + min(time_budget, max(0.5, _LIMIT - (time.time() - _start_time) - 5.0))
    
    best_tour = None
    best_len = float('inf')
    
    # Strategy 1: Nearest neighbor tour
    nn_tour = _nn_tour(distance_matrix, start_node)
    nn_tour = _two_opt(nn_tour, distance_matrix, max_passes=30)
    nn_len = _tour_length(nn_tour, distance_matrix)
    if nn_len < best_len:
        best_len = nn_len
        best_tour = list(nn_tour)
    
    if time.time() > deadline:
        return best_tour
    
    # Strategy 2: Greedy insertion tour
    gi_tour = _greedy_insertion_tour(distance_matrix, start_node)
    gi_tour = _two_opt(gi_tour, distance_matrix, max_passes=30)
    gi_len = _tour_length(gi_tour, distance_matrix)
    if gi_len < best_len:
        best_len = gi_len
        best_tour = list(gi_tour)
    
    if time.time() > deadline:
        return best_tour
    
    # Strategy 3: Multi-start NN with different starting points
    # Try a few different starting cities
    tried_starts = {start_node}
    # Pick a few diverse starting points
    candidate_starts = []
    for _ in range(min(5, n - 1)):
        # Pick a random-ish start (use deterministic selection based on index)
        s = (start_node + 1) % n
        if s not in tried_starts:
            candidate_starts.append(s)
            tried_starts.add(s)
    
    for s in candidate_starts:
        if time.time() > deadline:
            break
        ms_tour = _nn_tour(distance_matrix, s)
        # Ensure start_node is in the tour (it should be since we visit all nodes)
        ms_tour = _two_opt(ms_tour, distance_matrix, max_passes=20)
        ms_len = _tour_length(ms_tour, distance_matrix)
        if ms_len < best_len:
            best_len = ms_len
            best_tour = list(ms_tour)
    
    # Final 2-opt polish on the best tour found
    if best_tour is not None and time.time() < deadline - 0.5:
        best_tour = _two_opt(best_tour, distance_matrix, max_passes=50)
    
    return best_tour


def _detect_new_instance(unvisited_nodes, distance_matrix):
    """Detect if this is the first call for a new instance."""
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    return n_unvisited >= n_total - 1


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses a precomputed reference tour (built via multi-start NN + greedy insertion + 2-opt)
    and follows it. This is a structurally different approach from per-step scoring.
    """
    global _instance_tour, _instance_id_key, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    
    # Detect new instance
    is_new_instance = _detect_new_instance(unvisited_nodes, distance_matrix)
    
    if is_new_instance:
        _instance_counter += 1
        # Use more time for first instance, less for subsequent
        tb = 10.0 if _instance_counter == 1 else 6.0
        _instance_tour = _build_reference_tour(distance_matrix, current_node, time_budget=tb)
        # Create a key to identify this instance
        _instance_id_key = (n_total, current_node, tuple(sorted(unvisited_nodes)))
    
    # If no precomputed tour or deadline passed, fallback to nearest neighbor
    if _instance_tour is None or _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Find the next node in the precomputed tour
    # The tour is a sequence; we need to find which node comes after current_node
    # in the tour, considering only unvisited nodes
    
    tour = _instance_tour
    n_tour = len(tour)
    
    # Find current_node's position in the tour
    try:
        cur_pos = tour.index(current_node)
    except ValueError:
        # current_node not in tour (shouldn't happen), fallback to NN
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Get the set of unvisited nodes for quick lookup
    unvisited_set = set(int(u) for u in unvisited_nodes)
    
    # Find the next unvisited node in the tour after current_node
    for offset in range(1, n_tour + 1):
        next_pos = (cur_pos + offset) % n_tour
        next_node = tour[next_pos]
        if next_node in unvisited_set:
            return int(next_node)
    
    # Fallback: nearest neighbor
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    best_idx = np.argmin(dists_from_current)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
