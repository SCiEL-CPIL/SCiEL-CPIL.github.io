"""
TSP constructive heuristic: High-quality reference tour + smooth lookahead + exact tail.

Strategy:
  1. On detecting a new instance, build a high-quality reference tour using 
     Multi-Start NN + 2-Opt + Or-Opt.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction:
     a. If the reference successor is unvisited, follow it.
     b. If not, use a 3-step lookahead heuristic with smooth min (log-sum-exp)
        to select the next node.
     c. For small remaining sets (<=10), use exact brute force to optimize the tail.
  4. Deadline guard: if time is running out, fall back to nearest neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 145.0  # Safety margin before 150s limit

# Per-instance state
_last_dm_checksum = None
_reference_successor = None
_instance_initialized = False
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start):
    """Build a tour using nearest neighbor from a given start node."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour


def _tour_length(tour, distance_matrix):
    """Compute the total length of a tour."""
    n = len(tour)
    return sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _two_opt(tour, distance_matrix):
    """Apply 2-Opt local search to improve the tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if _is_deadline_passed():
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Check if reversing tour[i+1..j] improves the tour
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour


def _or_opt(tour, distance_matrix):
    """Apply Or-Opt local search (move 1, 2, or 3 consecutive nodes)."""
    n = len(tour)
    for segment_size in [1, 2, 3]:
        for i in range(n):
            if _is_deadline_passed():
                return tour
            segment = tour[i:i + segment_size] if i + segment_size <= n else tour[i:] + tour[:i + segment_size - n]
            if len(segment) < segment_size:
                continue
            # Remove segment from tour
            remaining = [x for x in tour if x not in segment]
            # Try inserting segment at each position
            best_insert = -1
            best_delta = 0
            for j in range(n):
                if j == i:
                    continue
                # Compute delta for inserting segment at position j
                # Before: remaining[j-1] -> remaining[j]
                # After: remaining[j-1] -> segment[0] ... segment[-1] -> remaining[j]
                if j == 0:
                    prev_node = remaining[-1]
                else:
                    prev_node = remaining[j - 1]
                next_node = remaining[j]
                
                # Cost before
                cost_before = distance_matrix[prev_node][next_node]
                # Cost after
                cost_after = distance_matrix[prev_node][segment[0]] + distance_matrix[segment[-1]][next_node]
                # Internal costs of segment (same in both)
                
                delta = cost_after - cost_before
                if delta < best_delta:
                    best_delta = delta
                    best_insert = j
            
            if best_insert >= 0 and best_delta < -1e-10:
                new_tour = remaining[:best_insert] + segment + remaining[best_insert:]
                tour = new_tour
    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a high-quality reference tour using Multi-Start NN + 2-Opt + Or-Opt.
    Returns an array where reference_successor[i] is the node visited after i in the tour.
    """
    global _reference_successor
    
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([], dtype=int)
    if n == 2:
        return np.array([1, 0])
    
    best_tour = None
    best_len = float('inf')
    
    # Multi-start: try all starting nodes
    for start in range(n):
        if _is_deadline_passed():
            break
        
        # Build NN tour from this start
        tour = _nn_tour(distance_matrix, start)
        
        # Apply 2-Opt
        tour = _two_opt(tour, distance_matrix)
        
        # Apply Or-Opt
        tour = _or_opt(tour, distance_matrix)
        
        # Compute tour length
        tour_len = _tour_length(tour, distance_matrix)
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour
    
    if best_tour is None:
        # Fallback: simple NN from node 0
        best_tour = _nn_tour(distance_matrix, 0)
    
    # Build successor array
    successor = np.zeros(n, dtype=int)
    for i in range(n):
        successor[best_tour[i]] = best_tour[(i + 1) % n]
    
    _reference_successor = successor
    return successor


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick checksum
        step = max(1, n_total // 10)
        checksum = np.sum(distance_matrix[::step, ::step])
        
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True
    
    return False


def _exact_tail_optimization(current_node, destination_node, unvisited_nodes, distance_matrix):
    """For small remaining sets, use exact brute force to optimize the tail."""
    nodes = list(unvisited_nodes)
    best_cost = np.inf
    best_first_node = nodes[0]
    
    # Limit permutations if n is slightly larger, but 10! is 3.6M which is too slow.
    # Use 8! = 40320 which is fast.
    for perm in permutations(nodes):
        # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
        cost = distance_matrix[current_node, perm[0]]
        for i in range(len(perm) - 1):
            cost += distance_matrix[perm[i], perm[i+1]]
        cost += distance_matrix[perm[-1], destination_node]
        
        if cost < best_cost:
            best_cost = cost
            best_first_node = perm[0]
            
    return int(best_first_node)


def _smooth_min(values, temperature=1.0):
    """
    Compute a smooth approximation of the minimum using log-sum-exp.
    As temperature -> 0, this approaches the hard min.
    """
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    min_val = np.min(values)
    # Avoid overflow: subtract min before exponentiating
    shifted = values - min_val
    log_sum_exp = min_val + np.log(np.sum(np.exp(-shifted / temperature)))
    # The soft min is -temperature * log(sum(exp(-values/temperature)))
    # But we want a smooth version of min, so:
    # soft_min = -temperature * log(sum(exp(-values / temperature)))
    # This is equivalent to min_val - temperature * log(sum(exp(-shifted/temperature)))
    return min_val - temperature * np.log(np.sum(np.exp(-shifted / temperature)))


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Follows the pre-computed reference tour's successor when possible,
    falling back to a smooth 3-step lookahead scoring rule when unavailable.
    """
    global _instance_initialized, _reference_successor, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If few nodes left, use exact brute force to optimize the tail
    # 8! = 40320 permutations, which is fast.
    if n_unvisited <= 8:
        return _exact_tail_optimization(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_initialized = False
        _reference_successor = None
        # Build reference tour
        _build_reference_tour(distance_matrix)
        _instance_initialized = True
        _instance_counter += 1
    
    if _is_deadline_passed():
        # Fallback: pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Try to follow the reference tour
    if _reference_successor is not None:
        ref_next = int(_reference_successor[current_node])
        # Check if ref_next is in unvisited_nodes
        if ref_next in unvisited_nodes:
            return ref_next
    
    # Fallback: 3-step lookahead scoring with smooth min
    # Score = dist(current, c) + smooth_min_{d in unvisited, d!=c} [dist(c, d) + smooth_min_{e in unvisited, e!=c,d} dist(d, e)]
    # Simplified: use 2-step with smooth min, plus destination bias
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Vectorized 2-step lookahead with smooth min
    # sub_dist is n_unvisited x n_unvisited
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dist, np.inf)
    
    # For each candidate c, compute smooth min of dist(c, d) over d != c
    temperature = 0.5  # Lower = closer to hard min
    min_next_dist = np.array([_smooth_min(sub_dist[i], temperature) for i in range(len(unvisited_nodes))])
    
    # Combine: immediate cost + penalty for future isolation
    # We also add a small bias towards destination to prevent ending up far from dest
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Tuned weights: immediate cost + 0.8 * future cost + 0.15 * destination bias
    scores = dists_from_current + 0.8 * min_next_dist + 0.15 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
