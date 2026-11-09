"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a connectivity-aware heuristic that balances:
    1. Distance from current node to candidate
    2. Distance from candidate to destination (lookahead)
    3. Penalty for candidates that would create long gaps / disconnected subtours
    
    Uses a combination of:
    - Direct edge cost (current -> candidate)
    - Lookahead edge cost (candidate -> destination)
    - MST-based lower bound on remaining path to penalize choices that
      fragment the remaining graph
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    
    # If only one node left, pick it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if close, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # For small numbers of unvisited nodes, we can afford more computation
    # Compute scores for each candidate
    scores = np.zeros(n_unvisited)
    
    for i, cand in enumerate(unvisited_nodes):
        # Direct cost: current -> candidate
        d_current_to_cand = distance_matrix[current_node, cand]
        
        # Lookahead cost: candidate -> destination
        d_cand_to_dest = distance_matrix[cand, destination_node]
        
        # MST-based lower bound: approximate cost of connecting all remaining
        # unvisited nodes (excluding candidate) plus candidate into a path
        # that starts after candidate and ends at destination
        
        if n_unvisited <= 1:
            mst_lb = 0.0
        else:
            # Get the set of nodes that would remain after picking candidate
            # These are all unvisited except candidate
            remaining_mask = np.ones(n_unvisited, dtype=bool)
            remaining_mask[i] = False
            remaining_nodes = unvisited_nodes[remaining_mask]
            
            # Build a small distance matrix for: {candidate} U remaining_nodes U {destination}
            # We want the MST lower bound for connecting candidate to destination
            # through the remaining nodes
            
            # For efficiency with small n, compute MST on the induced subgraph
            # Nodes: candidate, all remaining unvisited, destination
            special_nodes = np.concatenate(([cand], remaining_nodes, [destination_node]))
            n_special = len(special_nodes)
            
            if n_special <= 1:
                mst_lb = 0.0
            elif n_special <= 2:
                mst_lb = distance_matrix[special_nodes[0], special_nodes[1]]
            else:
                # Compute MST using Prim's algorithm on the small subgraph
                dm_sub = distance_matrix[np.ix_(special_nodes, special_nodes)]
                
                # Prim's algorithm
                in_mst = np.zeros(n_special, dtype=bool)
                min_dist = np.full(n_special, np.inf)
                min_dist[0] = 0.0
                mst_weight = 0.0
                
                for _ in range(n_special):
                    # Find unvisited node with minimum distance
                    candidates = np.where(~in_mst)[0]
                    if len(candidates) == 0:
                        break
                    min_idx = candidates[np.argmin(min_dist[candidates])]
                    mst_weight += min_dist[min_idx]
                    in_mst[min_idx] = True
                    # Update distances
                    for j in range(n_special):
                        if not in_mst[j] and dm_sub[min_idx, j] < min_dist[j]:
                            min_dist[j] = dm_sub[min_idx, j]
                
                mst_lb = mst_weight
        
        # Combine scores:
        # Primary: direct edge cost
        # Secondary: lookahead to destination
        # Tertiary: MST lower bound (penalizes choices that make remaining path expensive)
        
        # Weight the components
        w_direct = 1.0
        w_lookahead = 0.3
        w_mst = 0.2
        
        scores[i] = w_direct * d_current_to_cand + w_lookahead * d_cand_to_dest + w_mst * mst_lb
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
