"""
Hybrid TSP constructive heuristic:
  1. On detecting a new instance, build a high-quality reference tour using
     multi-start nearest-neighbor followed by 2-opt and Or-opt.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, follow the reference tour's successor when possible.
  4. If the current node's reference successor is already visited, fall back to
     a local scoring rule with a small 2-step lookahead and an isolation penalty.

This combines the "follow-the-best-tour" structural idea with the parameterized
scoring refinement from the current program.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_instance_id = -1
_reference_successor = None  # array: reference_successor[i] = next node after i in ref tour
_instance_initialized = False
_instance_dm = None


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
                seg_start = tour[i]
                seg_end = tour[(i + seg_len - 1) % n]
                before = tour[(i - 1) % n]
                after = tour[(i + seg_len) % n]

                # Try inserting at every position j
                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    if j == i or j == (i - 1) % n or j == (i + seg_len) % n:
                        continue
                    in_seg = False
                    for k in range(seg_len):
                        if (i + k) % n == j:
                            in_seg = True
                            break
                    if in_seg:
                        continue

                    # Compute delta for moving segment
                    # Original edges: (before, seg_start), (seg_end, after)
                    # New edges: (before, after), (tour[(j-1)%n], seg_start), (seg_end, tour[j%n])
                    # But we need to be careful about the tour structure
                    # For simplicity, compute the delta directly
                    insert_before = tour[(j - 1) % n]
                    insert_after = tour[j % n]

                    # If j is between i and i+seg_len, skip
                    if i < j < i + seg_len:
                        continue

                    delta = (distance_matrix[before][after] +
                             distance_matrix[insert_before][seg_start] +
                             distance_matrix[seg_end][insert_after] -
                             distance_matrix[before][seg_start] -
                             distance_matrix[seg_end][after])

                    if delta < -1e-10:
                        # Perform the move
                        # Remove segment from position i
                        seg = tour[i:i + seg_len]
                        tour = tour[:i] + tour[i + seg_len:]
                        # Insert at position j (adjust for removal)
                        if j > i:
                            j -= seg_len
                        tour = tour[:j] + seg + tour[j:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour


def _build_reference_tour(distance_matrix, start_node):
    """Build a high-quality reference tour using multi-start NN + 2-opt + Or-opt."""
    n = distance_matrix.shape[0]
    if n < 4:
        return list(range(n))

    best_tour = None
    best_len = float('inf')

    # Try a few random starts for diversity
    starts = [start_node]
    # Add a few random starts
    for _ in range(3):
        s = int(np.random.randint(0, n))
        if s not in starts:
            starts.append(s)

    for start in starts:
        if _is_deadline_passed():
            break
        tour = _nn_tour(distance_matrix, start)
        tour = _two_opt(distance_matrix, tour)
        tour = _or_opt(distance_matrix, tour)

        # Compute tour length
        tour_len = 0.0
        for i in range(len(tour)):
            a, b = tour[i], tour[(i + 1) % len(tour)]
            tour_len += distance_matrix[a][b]

        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour

    if best_tour is None:
        best_tour = list(range(n))

    return best_tour


def _compute_successor(tour):
    """Compute the successor of each node in the tour."""
    n = len(tour)
    successor = np.zeros(n, dtype=int)
    for i in range(n):
        successor[tour[i]] = tour[(i + 1) % n]
    return successor


def _score_candidate(current_node, destination_node, unvisited_nodes, distance_matrix,
                     reference_successor, alpha=0.1, gamma=0.05, delta=0.05):
    """
    Score candidates using:
      - distance from current
      - small weight to distance to destination
      - 2-step lookahead (min distance to other unvisited)
      - isolation penalty
      - bonus if candidate is the reference successor of current
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Check if reference successor is available
    ref_succ = reference_successor[current_node]
    ref_succ_in_unvisited = ref_succ in unvisited_nodes

    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()

    # Add small weight to distance to destination
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest

    # 2-step lookahead
    if gamma > 0 and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next

    # Isolation penalty
    if delta > 0 and n_unvisited > 2:
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)

    # If reference successor is in unvisited, give it a strong bonus
    if ref_succ_in_unvisited:
        idx = np.where(unvisited_nodes == ref_succ)[0][0]
        scores[idx] -= 100.0  # Strong bonus to follow reference tour

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.

    Uses a hybrid approach:
    1. On detecting a new instance, build a high-quality reference tour.
    2. Follow the reference tour's successor when possible.
    3. Fall back to a local scoring rule with lookahead and isolation penalty.
    """
    global _instance_id, _reference_successor, _instance_initialized, _instance_dm

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return -1

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Detect new instance
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)

    if is_new_instance:
        # New instance, build reference tour
        if _is_deadline_passed():
            # Fallback to nearest neighbor
            dists = distance_matrix[current_node, unvisited_nodes]
            best_idx = np.argmin(dists)
            return int(unvisited_nodes[best_idx])

        ref_tour = _build_reference_tour(distance_matrix, current_node)
        _reference_successor = _compute_successor(ref_tour)
        _instance_initialized = True
        _instance_dm = distance_matrix

    if _reference_successor is None:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Use the scoring function
    return _score_candidate(current_node, destination_node, unvisited_nodes,
                            distance_matrix, _reference_successor,
                            alpha=0.1, gamma=0.05, delta=0.05)

# EVOLVE-BLOCK-END
