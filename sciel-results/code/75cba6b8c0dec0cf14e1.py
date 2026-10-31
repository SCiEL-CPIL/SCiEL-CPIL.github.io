# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_last_dm_checksum = None
_reference_successor = None  # array: reference_successor[i] = next node after i in ref tour
_instance_initialized = False


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


def _greedy_insertion_tour(distance_matrix, start):
    """Build a tour using greedy insertion heuristic."""
    n = distance_matrix.shape[0]
    # Start with a small initial tour: start node and its two nearest neighbors
    dists_from_start = distance_matrix[start].copy()
    dists_from_start[start] = np.inf
    
    # Find the nearest neighbor to start
    nn1 = int(np.argmin(dists_from_start))
    dists_from_start[nn1] = np.inf
    nn2 = int(np.argmin(dists_from_start))
    
    # Initial tour: start -> nn1 -> nn2 (or just start -> nn1 if n <= 2)
    if n <= 2:
        tour = [start] + [nn1] if n > 1 else [start]
    else:
        # Choose initial 3-node tour that minimizes length
        # Options: start-nn1-nn2, start-nn2-nn1, nn1-start-nn2, etc.
        # For simplicity, use start -> nn1 -> nn2
        tour = [start, nn1, nn2]
    
    visited = np.zeros(n, dtype=bool)
    for node in tour:
        visited[node] = True
    
    unvisited = set(range(n)) - set(tour)
    
    # Greedy insertion: at each step, insert the unvisited node that can be added
    # with minimum additional cost into the current tour
    while len(tour) < n:
        if _is_deadline_passed():
            break
        best_insert_cost = np.inf
        best_node = None
        best_pos = None
        
        for node in unvisited:
            if _is_deadline_passed():
                break
            # Try inserting node at each position in the tour
            m = len(tour)
            for pos in range(m):
                # Insert after tour[pos], i.e., between tour[pos] and tour[(pos+1)%m]
                prev_node = tour[pos]
                next_node = tour[(pos + 1) % m]
                # Additional cost: dist(prev, node) + dist(node, next) - dist(prev, next)
                add_cost = distance_matrix[prev_node, node] + distance_matrix[node, next_node] - distance_matrix[prev_node, next_node]
                if add_cost < best_insert_cost:
                    best_insert_cost = add_cost
                    best_node = node
                    best_pos = pos
        
        if best_node is None:
            break
        
        # Insert best_node at best_pos
        tour.insert(best_pos + 1, best_node)
        visited[best_node] = True
        unvisited.remove(best_node)
    
    return tour


def _two_opt(distance_matrix, tour):
    """Improve a tour using 2-opt until no improvement is found."""
    n = len(tour)
    if n < 4:
        return tour

    tour = list(tour)
    improved = True

    while improved and not _is_deadline_passed():
        improved = False
        for i in range(n - 2):
            if _is_deadline_passed():
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(distance_matrix, tour):
    """Improve a tour using Or-Opt (relocate 1-3 city segments)."""
    n = len(tour)
    if n < 4:
        return tour

    tour = list(tour)
    improved = True

    while improved and not _is_deadline_passed():
        improved = False
        for seg_len in [1, 2, 3]:
            if _is_deadline_passed():
                return tour
            for i in range(n):
                if _is_deadline_passed():
                    return tour
                seg = tour[i:i + seg_len]
                if len(seg) < seg_len:
                    continue

                before = tour[(i - 1) % n]
                after = tour[(i + seg_len) % n]
                seg_start = seg[0]
                seg_end = seg[-1]

                # Try inserting at every other position
                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    # Skip positions that would recreate the same or adjacent
                    if j == i or j == (i - 1) % n or j == (i + seg_len) % n:
                        continue
                    # Skip if j is within the segment range
                    in_seg = False
                    for k in range(seg_len):
                        if (i + k) % n == j:
                            in_seg = True
                            break
                    if in_seg:
                        continue

                    insert_before = tour[(j - 1) % n]
                    insert_after = tour[j]

                    # Compute delta
                    # Remove edges: (before, seg_start), (seg_end, after)
                    # Add edges: (before, after), (insert_before, seg_start), (seg_end, insert_after)
                    # But we need to be careful about which edges are actually present
                    # For simplicity, compute the delta directly
                    delta = (distance_matrix[before][after] + distance_matrix[insert_before][seg_start] + distance_matrix[seg_end][insert_after]
                             - distance_matrix[before][seg_start] - distance_matrix[seg_end][after])
                    if delta < -1e-10:
                        # Perform the move
                        # Remove segment from position i
                        new_tour = tour[:i] + tour[i + seg_len:]
                        # Insert segment at position j (adjust for removal)
                        # If j > i, the index shifts by seg_len
                        if j > i:
                            insert_pos = j - seg_len
                        else:
                            insert_pos = j
                        new_tour = new_tour[:insert_pos] + seg + new_tour[insert_pos:]
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break
    return tour


def _tour_length(distance_matrix, tour):
    """Calculate the total length of a closed tour."""
    n = len(tour)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total


def _build_reference_tour(distance_matrix, start_node):
    """Build a high-quality reference tour and compute successor mapping."""
    global _reference_successor, _last_dm_checksum, _instance_initialized

    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf

    # Multi-Start with both NN and Greedy Insertion
    # Try a few start nodes to get a better initial tour
    start_candidates = list(range(min(10, n)))
    # Ensure the original start_node is included
    if start_node not in start_candidates:
        start_candidates.append(start_node)
        
    for s in start_candidates:
        if _is_deadline_passed():
            break
        
        # Try nearest neighbor
        tour = _nn_tour(distance_matrix, s)
        tour = _two_opt(distance_matrix, tour)
        length = _tour_length(distance_matrix, tour)
        if length < best_len:
            best_len = length
            best_tour = tour
        
        # Try greedy insertion if time permits
        if not _is_deadline_passed():
            tour_gi = _greedy_insertion_tour(distance_matrix, s)
            tour_gi = _two_opt(distance_matrix, tour_gi)
            length_gi = _tour_length(distance_matrix, tour_gi)
            if length_gi < best_len:
                best_len = length_gi
                best_tour = tour_gi
            
    if best_tour is None:
        # Fallback if time ran out before any tour was built (unlikely)
        best_tour = _nn_tour(distance_matrix, start_node)
        best_tour = _two_opt(distance_matrix, best_tour)

    # Apply Or-Opt to the best found tour
    if not _is_deadline_passed():
        best_tour = _or_opt(distance_matrix, best_tour)

    # Compute successor mapping
    _reference_successor = np.zeros(n, dtype=int)
    for i in range(n):
        _reference_successor[best_tour[i]] = best_tour[(i + 1) % n]

    # Store checksum to detect new instances
    _last_dm_checksum = hash(tuple(tuple(row) for row in distance_matrix))
    _instance_initialized = True

    return best_tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.

    Uses reference-tour guidance: builds a high-quality reference tour on detecting
    a new instance, then follows the reference tour's successor when possible.
    Falls back to local scoring if the reference successor is already visited.
    """
    global _last_dm_checksum, _reference_successor, _instance_initialized

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return -1

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # If few nodes left, use exact brute force to optimize the tail
    if n_unvisited <= 5:
        nodes = list(unvisited_nodes)
        best_cost = np.inf
        best_first_node = nodes[0]

        for perm in permutations(nodes):
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i + 1]]
            cost += distance_matrix[perm[-1], destination_node]

            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]

        return int(best_first_node)

    # Detect new instance
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)  # First or second step of a new tour

    if is_new_instance:
        # Check if we've already built a reference for this instance
        current_checksum = hash(tuple(tuple(row) for row in distance_matrix))
        if _last_dm_checksum != current_checksum:
            _build_reference_tour(distance_matrix, current_node)

    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Try to follow the reference tour's successor
    if _reference_successor is not None:
        ref_next = _reference_successor[current_node]
        if ref_next in unvisited_nodes:
            return int(ref_next)

    # Fallback: local scoring rule with spatial awareness
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()

    # Small destination pull
    alpha = 0.05
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    scores += alpha * dists_to_dest

    # 2-step lookahead
    if n_unvisited > 2:
        gamma = 0.1
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next

    # Spatial decomposition penalty: penalize candidates that are isolated
    # (far from other unvisited nodes) to prevent creating isolated clusters
    if n_unvisited > 3:
        # Compute for each unvisited node, the distance to its nearest other unvisited node
        sub_dist_full = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist_full, np.inf)
        nearest_dist = np.min(sub_dist_full, axis=1)
        
        # Normalize: nodes with larger nearest-neighbor distance are more isolated
        # Use a soft penalty: scale by median nearest distance
        median_nn = np.median(nearest_dist)
        if median_nn > 1e-10:
            iso_penalty = 0.02 * (nearest_dist / median_nn)
            scores += iso_penalty

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
