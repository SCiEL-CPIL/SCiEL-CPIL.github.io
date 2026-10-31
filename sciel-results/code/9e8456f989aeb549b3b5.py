"""Traveling salesman constructive heuristic with lookahead and destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = 120.0  # seconds
_START_TIME = None


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour construction.
    
    Strategy: Combine nearest-neighbor with a small lookahead (2-step) 
    and a bias toward the destination to reduce edge crossings and 
    produce shorter tours.
    
    We score each candidate unvisited node by:
      - Primary: distance from current to candidate
      - Secondary: distance from candidate to destination (bias)
      - Lookahead: for the top-k candidates by primary score, check 
        the 2-step path cost (current->cand->nearest_unvisited_after)
    
    This is a refinement of the nearest-neighbor baseline, staying 
    within the same constructive heuristic family but adding 
    destination-awareness and limited lookahead.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety: if we're running out of time, fall back to pure NN
    if time.time() - _START_TIME > _DEADLINE - 2.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from destination to all unvisited (for bias)
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Score: primarily distance from current, with a small bias toward destination
    # Using a weighted combination: 0.8 * dist_from_current + 0.2 * dist_to_dest
    # This gently pulls the tour toward the destination without being too aggressive
    scores = 0.8 * dist_from_current + 0.2 * dist_to_dest
    
    # Get top-k candidates by score (small k for efficiency)
    k = min(5, n_unvisited)
    top_k_indices = np.argsort(scores)[:k]
    
    # For each top candidate, compute a 2-step lookahead cost
    # Lookahead: current -> candidate -> (best next unvisited after removing candidate)
    # This helps avoid local traps where the nearest neighbor leads to a long detour
    best_candidate = top_k_indices[0]
    best_lookahead_cost = np.inf
    
    for idx in top_k_indices:
        cand = unvisited_nodes[idx]
        
        # Cost of current -> cand
        step1_cost = distance_matrix[current_node, cand]
        
        # For step 2: find the best unvisited node after removing cand
        # We approximate by looking at the minimum distance from cand to any other unvisited
        # (excluding cand itself)
        if n_unvisited > 1:
            # Create a mask to exclude the candidate
            other_unvisited = np.delete(unvisited_nodes, idx)
            dist_from_cand_to_others = distance_matrix[cand, other_unvisited]
            min_step2_cost = np.min(dist_from_cand_to_others)
        else:
            min_step2_cost = 0.0
        
        # Total lookahead cost
        lookahead_cost = step1_cost + 0.5 * min_step2_cost
        
        if lookahead_cost < best_lookahead_cost:
            best_lookahead_cost = lookahead_cost
            best_candidate = idx
    
    return int(unvisited_nodes[best_candidate])
# EVOLVE-BLOCK-END
