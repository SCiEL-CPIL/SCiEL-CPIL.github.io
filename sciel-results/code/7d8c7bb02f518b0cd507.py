"""
TSP Constructive Heuristic: Reference-Tour Guided Greedy.

Strategy:
1. Pre-compute a high-quality reference tour using Multi-Start Nearest Neighbor 
   followed by 2-Opt local search.
2. During constructive selection, follow the successor of the current node in 
   the reference tour if possible.
3. If the reference path is not available, use a greedy heuristic that balances 
   proximity to the current node and proximity to the destination (start node).
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 110.0  # Safety margin before 120s limit

# Per-instance state
_ref_successor = {}
_instance_detected = False

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def _nn_tour(distance_matrix, start_node):
    """Generate a tour using Nearest Neighbor heuristic."""
    n = distance_matrix.shape[0]
    current = start_node
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    
    while unvisited:
        # Find nearest neighbor
        dists = distance_matrix[current]
        # Mask visited nodes
        min_dist = float('inf')
        next_node = -1
        for node in unvisited:
            if dists[node] < min_dist:
                min_dist = dists[node]
                next_node = node
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    # Close the tour
    tour.append(start_node)
    return tour

def _two_opt(tour, distance_matrix):
    """Improve a tour using 2-Opt local search."""
    n = len(tour) - 1 # Exclude closing node
    improved = True
    # Limit iterations to prevent infinite loops or excessive time
    max_iter = 1000
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            if _is_deadline_passed():
                return tour
            for j in range(i + 1, n):
                # Check if edges (i-1, i) and (j, j+1) can be swapped
                # New edges would be (i-1, j) and (i, j+1)
                # Old edges: (i-1, i) and (j, j+1)
                # Note: tour[k] is the k-th node. tour[0] is start.
                # Edge 1: tour[i-1] to tour[i]
                # Edge 2: tour[j] to tour[j+1]
                
                d_old = distance_matrix[tour[i-1], tour[i]] + distance_matrix[tour[j], tour[j+1]]
                d_new = distance_matrix[tour[i-1], tour[j]] + distance_matrix[tour[i], tour[j+1]]
                
                if d_new < d_old - 1e-10:
                    # Reverse the segment between i and j
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def _build_reference_tour(distance_matrix):
    """Build a high-quality reference tour using multi-start NN + 2-Opt."""
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = float('inf')
    
    # Try a few different starting nodes
    # For 50 nodes, trying all 50 starts with 2-Opt is feasible within time limits
    # but we can start with a subset if time is tight.
    # Let's try all starts but break early if deadline is close.
    
    for start in range(n):
        if _is_deadline_passed():
            break
            
        nn_tour = _nn_tour(distance_matrix, start)
        # Calculate length of NN tour
        tour_len = sum(distance_matrix[nn_tour[i], nn_tour[i+1]] for i in range(len(nn_tour)-1))
        
        # Apply 2-Opt
        opt_tour = _two_opt(nn_tour.copy(), distance_matrix)
        
        # Calculate length of optimized tour
        opt_len = sum(distance_matrix[opt_tour[i], opt_tour[i+1]] for i in range(len(opt_tour)-1))
        
        if opt_len < best_len:
            best_len = opt_len
            best_tour = opt_tour
            
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    """
    global _ref_successor, _instance_detected
    
    n_unvisited = len(unvisited_nodes)
    n_total = distance_matrix.shape[0]
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    if n_unvisited == n_total:
        _instance_detected = True
        _ref_successor = {}
        
        # Pre-compute reference tour
        if not _is_deadline_passed():
            ref_tour = _build_reference_tour(distance_matrix)
            if ref_tour:
                # Map each node to its successor in the reference tour
                for i in range(len(ref_tour) - 1):
                    _ref_successor[ref_tour[i]] = ref_tour[i+1]
    
    # If we have a reference successor for the current node and it's unvisited, use it
    if current_node in _ref_successor:
        next_node = _ref_successor[current_node]
        if next_node in unvisited_nodes:
            return int(next_node)
    
    # Fallback: Greedy with destination awareness
    # Score = d(current, c) + beta * d(c, destination)
    # This encourages moving towards the start to close the loop efficiently.
    
    if _is_deadline_passed():
        # Pure NN fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Heuristic weight: as we visit more nodes, we should care more about closing the loop
    # However, a fixed weight of 0.5 or 1.0 often works well for uniform instances.
    # Let's use 0.5 to balance immediate cost and future closing cost.
    beta = 0.5
    scores = dists_from_current + beta * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
