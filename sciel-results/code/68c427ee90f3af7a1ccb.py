"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour using a lookahead-based heuristic.
    
    The heuristic considers:
    1. Immediate distance from current_node to candidate
    2. Estimated cost of connecting candidate to the next-best unvisited node
    3. Cost of returning to destination (if candidate is last unvisited)
    
    This is a 2-step lookahead that balances immediate cost with future connectivity.
    """
    _ensure_start_time()
    
    if _time_exceeded():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, compute a lookahead score:
    # score(candidate) = dist(current, candidate) + min(dist(candidate, other_unvisited) for other != candidate)
    # If candidate is the last one, add dist(candidate, destination)
    
    # We want to minimize the total expected cost
    
    # Compute for each candidate i:
    #   immediate_cost[i] = dist_from_current[i]
    #   future_cost[i] = min over j != i of dist(candidate_i, candidate_j), 
    #                    or dist(candidate_i, destination) if only one left after this
    
    # For efficiency, we can compute the minimum outgoing distance for each candidate
    # to other unvisited nodes (excluding itself)
    
    # dist_matrix among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i, find the minimum distance to any other node j != i
    # Set diagonal to inf, then take row min
    np.fill_diagonal(sub_dist, np.inf)
    
    if n_unvisited > 1:
        min_outgoing = np.min(sub_dist, axis=1)
    else:
        min_outgoing = np.array([0.0])
    
    # For the last unvisited node, the "future" cost is distance to destination
    # We need to handle the case where choosing a candidate leaves exactly one unvisited
    # In that case, the cost is dist(candidate, last_remaining)
    
    # A simpler and effective approach: 
    # Use a weighted combination:
    # total_score[i] = dist_from_current[i] + alpha * min_outgoing[i]
    # where alpha is a tuning parameter (e.g., 0.5)
    
    # But we also need to consider the return to destination for the last node.
    # Let's use a more principled 2-step lookahead:
    
    # For each candidate i:
    #   If n_unvisited == 2:
    #     cost = dist(current, i) + dist(i, j) + dist(j, destination)  where j is the other
    #   If n_unvisited > 2:
    #     cost ≈ dist(current, i) + min_outgoing[i]  (greedy approximation)
    
    if n_unvisited == 2:
        # Exact 2-step lookahead
        i0, i1 = unvisited_nodes
        cost0 = distance_matrix[current_node, i0] + distance_matrix[i0, i1] + distance_matrix[i1, destination_node]
        cost1 = distance_matrix[current_node, i1] + distance_matrix[i1, i0] + distance_matrix[i0, destination_node]
        if cost0 <= cost1:
            return int(i0)
        else:
            return int(i1)
    else:
        # Use weighted score: immediate + future
        # Future cost is the minimum distance from candidate to any other unvisited node
        # This encourages choosing nodes that are well-connected to the rest
        
        # We also want to avoid getting stuck far from destination
        # Add a small penalty for being far from destination if few nodes remain
        
        score = dist_from_current + 0.5 * min_outgoing
        
        # If few nodes remain, weight the destination return more heavily
        if n_unvisited <= 5:
            dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
            # Blend in destination proximity
            score = score + 0.3 * dist_to_dest
        
        idx = np.argmin(score)
        return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
