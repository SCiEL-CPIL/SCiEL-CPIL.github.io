"""TSP constructive heuristic: build greedy tour, 2-opt optimize, then follow."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = None
_DEADLINE = 145.0
# Cache: keyed by (tuple of all nodes from first call) -> optimized tour
_tour_cache = {}
_started = False


def _remaining_time():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return _DEADLINE - (time.time() - _T0)


def _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix):
    if unvisited_nodes.size == 0:
        return int(current_node)
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])


def _build_greedy_tour(all_nodes, distance_matrix):
    """Nearest-neighbor tour starting from the first node."""
    n = all_nodes.size
    if n <= 1:
        return all_nodes.copy()
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=int)
    tour[0] = all_nodes[0]
    visited[0] = True
    cur = all_nodes[0]
    for i in range(1, n):
        d = distance_matrix[cur, all_nodes]
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        tour[i] = all_nodes[nxt]
        visited[nxt] = True
        cur = all_nodes[nxt]
    return tour


def _two_opt(tour, distance_matrix, max_passes=200):
    """2-opt improvement on a tour (list of node indices)."""
    n = len(tour)
    if n <= 3:
        return tour
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if _remaining_time() < 2.0:
            break
        for i in range(1, n - 1):
            if _remaining_time() < 2.0:
                return tour
            a = tour[i - 1]
            b = tour[i]
            for j in range(i + 1, n):
                c = tour[j - 1]
                d = tour[j]
                # Check if reversing tour[i..j] improves
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                if new_cost < old_cost - 1e-10:
                    # Reverse segment tour[i..j]
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    # After reversal, re-check from position i
                    break
    return tour


def _or_opt(tour, distance_matrix, max_passes=50):
    """Or-opt: move segments of length 1, 2, 3 to better positions."""
    n = len(tour)
    if n <= 4:
        return tour
    for seg_len in [1, 2, 3]:
        for _ in range(max_passes):
            improved = False
            if _remaining_time() < 2.0:
                return tour
            for i in range(n - seg_len):
                if _remaining_time() < 2.0:
                    return tour
                seg = tour[i:i+seg_len]
                # Remove segment
                # Find best insertion point
                best_gain = 1e-10
                best_pos = -1
                # Try inserting at each position j (after removing, indices shift)
                for j in range(n - seg_len + 1):
                    if j == i or j == i - 1:
                        continue
                    # Position j in the original tour (before removal)
                    # After removing seg at i, position j shifts
                    if j > i:
                        j_adj = j - seg_len
                    else:
                        j_adj = j
                    # Check if j_adj is valid and not adjacent to i
                    if j_adj < 0 or j_adj >= n - seg_len:
                        continue
                    # Skip if insertion is adjacent to removal
                    if j_adj == i or j_adj == i - 1:
                        continue
                    # Compute cost change
                    # Before removal: edges around i and around j
                    if i > 0 and i + seg_len < n:
                        old1 = distance_matrix[tour[i-1], tour[i]] + distance_matrix[tour[i+seg_len-1], tour[i+seg_len]]
                    elif i == 0:
                        old1 = distance_matrix[tour[i+seg_len-1], tour[i+seg_len]] if i + seg_len < n else distance_matrix[tour[n-1], tour[0]]
                    else:
                        old1 = distance_matrix[tour[i-1], tour[i]]
                    
                    if j_adj > 0 and j_adj < n - seg_len:
                        old2 = distance_matrix[tour[j_adj-1], tour[j_adj]]
                    elif j_adj == 0:
                        old2 = distance_matrix[tour[n-1], tour[0]]
                    else:
                        old2 = 0.0
                    
                    # New edges after insertion
                    if i > 0 and i + seg_len < n:
                        new1 = distance_matrix[tour[i-1], tour[i+seg_len]]
                    elif i == 0:
                        new1 = distance_matrix[tour[n-1], tour[i+seg_len]] if i + seg_len < n else distance_matrix[tour[n-1], tour[0]]
                    else:
                        new1 = 0.0
                    
                    if j_adj > 0 and j_adj < n - seg_len:
                        new2 = distance_matrix[tour[j_adj-1], seg[0]] + distance_matrix[seg[-1], tour[j_adj]]
                    elif j_adj == 0:
                        new2 = distance_matrix[tour[n-1], seg[0]] + distance_matrix[seg[-1], tour[0]]
                    else:
                        new2 = 0.0
                    
                    gain = (old1 + old2) - (new1 + new2)
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = j_adj
                
                if best_pos >= 0:
                    # Perform the move
                    seg = tour[i:i+seg_len].copy()
                    # Remove segment
                    tour = np.delete(tour, slice(i, i+seg_len))
                    # Insert at best_pos
                    tour = np.insert(tour, best_pos, seg)
                    improved = True
            if not improved:
                break
    return tour


def _compute_tour_length(tour, distance_matrix):
    n = len(tour)
    if n <= 1:
        return 0.0
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i+1) % n]]
    return total


def _optimize_tour(tour, distance_matrix):
    """Full optimization: 2-opt + or-opt."""
    tour = list(tour)
    tour = _two_opt(tour, distance_matrix, max_passes=300)
    tour = _or_opt(np.array(tour), distance_matrix, max_passes=30)
    # Another round of 2-opt
    tour = _two_opt(tour, distance_matrix, max_passes=100)
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _started
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = unvisited_nodes.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # On first call, build and optimize the full tour
    if not _started:
        _started = True
        # All nodes are the union of current_node, destination_node, and unvisited_nodes
        all_nodes = np.unique(np.concatenate([[current_node], [destination_node], unvisited_nodes]))
        cache_key = (tuple(all_nodes.tolist()),)
        
        if cache_key in _tour_cache:
            optimized_tour = _tour_cache[cache_key]
        else:
            # Build greedy tour
            greedy_tour = _build_greedy_tour(all_nodes, distance_matrix)
            # Optimize
            optimized_tour = _optimize_tour(greedy_tour, distance_matrix)
            _tour_cache[cache_key] = optimized_tour
        
        # Find current_node in the optimized tour and return the next node
        tour = optimized_tour
        try:
            idx = int(np.where(tour == current_node)[0][0])
            next_idx = (idx + 1) % len(tour)
            next_node = int(tour[next_idx])
            # Only return if next_node is in unvisited_nodes
            if next_node in unvisited_nodes:
                return next_node
        except (IndexError, ValueError):
            pass
        
        # Fallback: nearest neighbor
        return _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix)

    # Subsequent calls: follow the cached tour
    cache_key = (tuple(np.unique(np.concatenate([[current_node], [destination_node], unvisited_nodes])).tolist()),)
    if cache_key in _tour_cache:
        tour = _tour_cache[cache_key]
        try:
            idx = int(np.where(tour == current_node)[0][0])
            next_idx = (idx + 1) % len(tour)
            next_node = int(tour[next_idx])
            if next_node in unvisited_nodes:
                return next_node
        except (IndexError, ValueError):
            pass

    # Fallback: nearest neighbor
    return _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
