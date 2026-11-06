"""Traveling salesman constructive heuristic with local density (nearest-neighbor gap) scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin for 120s/150s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    # If only one node left, must pick it.
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # If two nodes left, pick the one that gives shorter closing
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        start_node = _current_tour[0]
        # Option 1: a then b, then back to start
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        # Option 2: b then a, then back to start
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    n_cands = len(unvisited_nodes)
    
    # 1. Immediate costs from current_node to all candidates
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Local density metric: for each candidate u, compute the "nearest-neighbor gap"
    # This measures how isolated u is from the rest of the unvisited cluster.
    # A small gap means u is well-integrated (close to other unvisited nodes),
    # which suggests visiting u now keeps the tour locally dense.
    # A large gap means u is an outlier; visiting it now may create a long detour.
    
    # Compute pairwise distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Mask diagonal with inf so we don't consider self-distance
    np.fill_diagonal(sub_matrix, np.inf)
    
    # For each candidate u, the nearest-neighbor gap is the minimum distance
    # from u to any other unvisited node.
    nn_gaps = np.min(sub_matrix, axis=1)
    
    # 3. Also compute the second-nearest neighbor gap for a more robust density measure.
    # Sort distances for each candidate and take the second smallest.
    sorted_dists = np.sort(sub_matrix, axis=1)
    second_nn_gaps = sorted_dists[:, 1]
    
    # 4. Composite score combining:
    #    - Immediate cost (proximity to current node)
    #    - Nearest-neighbor gap (local integration)
    #    - Second-nearest gap (cluster density)
    #    - Destination proximity (bias toward completing the tour)
    
    # Destination proximity: how close is each candidate to the destination?
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize components to comparable scales
    # Use median as a robust scale reference
    imm_scale = np.median(imm_costs) if n_cands > 0 else 1.0
    gap_scale = np.median(nn_gaps) if n_cands > 0 else 1.0
    dest_scale = np.median(dest_costs) if n_cands > 0 else 1.0
    
    # Avoid division by zero
    imm_scale = max(imm_scale, 1e-10)
    gap_scale = max(gap_scale, 1e-10)
    dest_scale = max(dest_scale, 1e-10)
    
    # Weighted composite score
    # Primary: immediate cost (greedy component)
    # Secondary: nn_gap (prefer well-integrated nodes)
    # Tertiary: destination proximity (bias toward closing the tour)
    w_imm = 1.0
    w_gap = 0.3
    w_dest = 0.1
    
    scores = (w_imm * (imm_costs / imm_scale) + 
              w_gap * (nn_gaps / gap_scale) + 
              w_dest * (dest_costs / dest_scale))
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    best_candidate = int(unvisited_nodes[best_idx])
    
    # Update global state
    _current_tour.append(best_candidate)
    
    return best_candidate

# EVOLVE-BLOCK-END
