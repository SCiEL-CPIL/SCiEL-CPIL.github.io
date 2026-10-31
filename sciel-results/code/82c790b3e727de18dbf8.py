"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global state for caching the pre-computed tour
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit
_cached_tour = None
_fingerprint = None
_current_step = 0

def _get_fingerprint(distance_matrix):
    """Create a simple fingerprint to identify the instance."""
    # Use a subset of values to create a hashable key
    # This is fast and sufficient for identifying the specific 50-city instance
    return tuple(distance_matrix[0, :10]) + tuple(distance_matrix[10, :10]) + tuple(distance_matrix[20, :10])

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the tour."""
    n = len(tour)
    if n == 0:
        return 0
    total = 0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total

def _nearest_neighbor_tour(distance_matrix):
    """Generate a baseline tour using Nearest Neighbor heuristic."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = []
    current = 0
    visited[0] = True
    tour.append(0)
    
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return np.array(tour)

def _two_opt(tour, distance_matrix, max_iterations=1000):
    """Improve tour using 2-Opt local search."""
    n = len(tour)
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for 2-opt move
                # Remove edges (i-1, i) and (j, j+1), add (i-1, j) and (i, j+1)
                # Indices in tour: 
                # prev_i = (i-1)%n, curr_i = i
                # next_j = (j+1)%n, curr_j = j
                prev_i = (i - 1) % n
                curr_i = i
                curr_j = j
                next_j = (j + 1) % n
                
                node_prev_i = tour[prev_i]
                node_curr_i = tour[curr_i]
                node_curr_j = tour[curr_j]
                node_next_j = tour[next_j]
                
                old_cost = distance_matrix[node_prev_i, node_curr_i] + distance_matrix[node_curr_j, node_next_j]
                new_cost = distance_matrix[node_prev_i, node_curr_j] + distance_matrix[node_curr_i, node_next_j]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse the segment between i and j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    
    return tour

def _or_opt(tour, distance_matrix, max_iterations=1000):
    """Improve tour using Or-Opt local search (moves segments of size 1, 2, 3)."""
    n = len(tour)
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for seg_len in [1, 2, 3]:
            for i in range(n):
                # Extract segment
                seg_nodes = [tour[(i + k) % n] for k in range(seg_len)]
                prev_node = tour[(i - 1) % n]
                next_node = tour[(i + seg_len) % n]
                
                # Try inserting this segment at every other position
                for j in range(n):
                    if j >= i and j < i + seg_len:
                        continue
                    # Check if insertion is valid (not adjacent)
                    if abs(i - j) <= seg_len or abs(i - j) >= n - seg_len:
                        continue
                    
                    # Calculate cost of removing seg and inserting at j
                    # Original edges: (prev, seg[0]), (seg[-1], next)
                    # New edges: (prev, next), (j_prev, seg[0]), (seg[-1], j_next)
                    
                    j_prev_node = tour[(j - 1) % n]
                    j_next_node = tour[j]
                    
                    old_cost = distance_matrix[prev_node, seg_nodes[0]] + distance_matrix[seg_nodes[-1], next_node]
                    new_cost = distance_matrix[prev_node, next_node] + distance_matrix[j_prev_node, seg_nodes[0]] + distance_matrix[seg_nodes[-1], j_next_node]
                    
                    if new_cost < old_cost - 1e-10:
                        # Perform move: remove seg and insert at j
                        # We need to be careful with indices when modifying the array
                        # Easier to reconstruct the list
                        tour_list = list(tour)
                        # Remove segment
                        del tour_list[i:i+seg_len]
                        # Insert at j (adjust index if j > i)
                        insert_idx = j
                        if j > i:
                            insert_idx = j - seg_len
                        # If j == i, we are moving to the same place, skip (handled by continue above)
                        # If j < i, insert at j
                        # If j > i, insert at j - seg_len
                        # Wait, if we delete from i, indices after i shift left.
                        # So if original j > i, new j is j - seg_len.
                        
                        # Let's just use a list copy for safety in this heuristic
                        new_tour_list = list(tour)
                        seg = new_tour_list[i:i+seg_len]
                        del new_tour_list[i:i+seg_len]
                        
                        # Find insertion point in the shortened list
                        # The original index j in the full list corresponds to...
                        # If j < i, it stays j.
                        # If j > i, it becomes j - seg_len.
                        # But we also need to consider if j is right after the segment (adjacent), which we skipped.
                        
                        if j > i:
                            insert_pos = j - seg_len
                        else:
                            insert_pos = j
                            
                        # If insert_pos is such that we are re-inserting adjacent to where it was, skip
                        # But we already checked adjacency in the full list context.
                        
                        new_tour_list[insert_pos:insert_pos] = seg
                        new_tour = np.array(new_tour_list)
                        
                        # Verify improvement
                        if _tour_length(new_tour, distance_matrix) < _tour_length(tour, distance_matrix) - 1e-10:
                            tour = new_tour
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return tour

def _compute_best_tour(distance_matrix):
    """
    Pre-compute a high-quality tour using NN + 2-Opt + Or-Opt.
    """
    global _cached_tour
    start = time.time()
    
    # 1. Generate initial tour
    tour = _nearest_neighbor_tour(distance_matrix)
    
    # 2. Apply 2-Opt
    tour = _two_opt(tour, distance_matrix)
    
    # 3. Apply Or-Opt
    tour = _or_opt(tour, distance_matrix)
    
    # 4. Another round of 2-Opt to polish
    tour = _two_opt(tour, distance_matrix)
    
    _cached_tour = tour
    
    # Check time
    elapsed = time.time() - start
    if elapsed > 10:
        print(f"Warning: Precomputation took {elapsed}s")

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by following a pre-computed optimal tour.
    """
    global _cached_tour, _fingerprint, _current_step
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Nearest neighbor
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Check if we have a cached tour for this instance
    fp = _get_fingerprint(distance_matrix)
    if _fingerprint != fp or _cached_tour is None:
        _fingerprint = fp
        _cached_tour = None
        _current_step = 0
        _compute_best_tour(distance_matrix)
        
    # If we still don't have a tour (e.g. timeout during computation), fallback
    if _cached_tour is None:
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Find the next node in the cached tour
    # We need to find the position of current_node in the tour
    # Then pick the next node in the tour that is in unvisited_nodes
    
    # Optimization: Pre-compute the order of nodes in the tour
    # tour_order[node] = index in tour
    # But we can just search. N=50 is small.
    
    # Find current index
    try:
        curr_idx = np.where(_cached_tour == current_node)[0][0]
    except IndexError:
        # Fallback if current node not in tour (should not happen)
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Iterate through the tour starting from curr_idx + 1
    n = len(_cached_tour)
    for k in range(1, n):
        next_node_in_tour = _cached_tour[(curr_idx + k) % n]
        if next_node_in_tour in unvisited_nodes:
            return int(next_node_in_tour)
            
    # If no unvisited node found in the tour (should not happen if tour is valid), fallback
    if len(unvisited_nodes) == 0:
        return int(current_node)
    dists = distance_matrix[current_node, unvisited_nodes]
    best_idx = np.argmin(dists)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
