"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track time and cache
_start_time = None
_time_limit = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()


def _time_exceeded():
    global _start_time
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _time_limit


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a traveling salesman tour.
    
    Uses a lookahead heuristic (2-opt lookahead) to improve upon simple nearest neighbor.
    For each candidate next node, we estimate the cost of going to that node and then
    to the best subsequent node (or destination if it's the last). We pick the candidate
    that minimizes this estimated cost.
    
    As a fallback for time pressure, we use nearest neighbor.
    """
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small numbers of unvisited nodes, we can afford more computation
    # Use a lookahead heuristic: for each candidate, estimate the total cost
    # of going current -> candidate -> best_next (or destination if last)
    
    # Get distances from current to all unvisited
    dist_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # If there are only 2 unvisited, we need to decide which to visit first
    # Cost of visiting i then j: dist(current,i) + dist(i,j) + dist(j,destination)
    # Cost of visiting j then i: dist(current,j) + dist(j,i) + dist(i,destination)
    # Since dist is symmetric, dist(i,j) = dist(j,i)
    # So we compare: dist(current,i) + dist(j,destination) vs dist(current,j) + dist(i,destination)
    
    if n_unvisited == 2:
        i, j = unvisited_nodes
        cost_ij = dist_to_unvisited[0] + distance_matrix[j, destination_node]
        cost_ji = dist_to_unvisited[1] + distance_matrix[i, destination_node]
        if cost_ij <= cost_ji:
            return int(i)
        else:
            return int(j)
    
    # For larger n_unvisited, use a 1-step lookahead
    # For each candidate i, estimate cost as:
    #   dist(current, i) + min(dist(i, j) for j in unvisited, j != i)
    # But this is expensive. Instead, use a simpler heuristic:
    #   dist(current, i) + alpha * dist(i, destination)
    # where alpha is a weighting factor. This biases towards nodes that are
    # not only close to current but also lead towards the destination.
    
    # Actually, let's use a proper 2-step lookahead but efficiently:
    # For each candidate i, find the best next node j (excluding i) and compute
    # dist(current, i) + dist(i, j)
    # Then pick i that minimizes this.
    
    # This is O(n^2) where n is the number of unvisited nodes.
    # For 50 cities, this is at most 2500 operations per step, which is fine.
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    for idx, candidate in enumerate(unvisited_nodes):
        # Cost from current to candidate
        cost_to_candidate = dist_to_unvisited[idx]
        
        # If this is the last but one, the next is the last unvisited
        # For lookahead, we estimate the cost of going to the best subsequent node
        # Exclude the candidate itself from the subsequent choices
        
        if n_unvisited == 2:
            # Already handled above
            pass
        
        # Find the best next node after candidate
        # We want min(dist(candidate, j)) for j in unvisited, j != candidate
        # But we should also consider the destination if it's the last step
        
        # For simplicity, use the minimum distance from candidate to any other unvisited
        # This is an approximation of the future cost
        
        # Get distances from candidate to all unvisited except itself
        dist_from_candidate = distance_matrix[candidate, unvisited_nodes]
        dist_from_candidate[idx] = np.inf  # Exclude self
        
        best_next_cost = np.min(dist_from_candidate)
        
        # Total estimated cost
        total_cost = cost_to_candidate + best_next_cost
        
        # Add a small bias towards the destination for the last few nodes
        # to ensure we don't get stuck far from the destination
        if n_unvisited <= 5:
            # When few nodes remain, consider the cost to destination
            dist_to_dest = distance_matrix[candidate, destination_node]
            total_cost += 0.5 * dist_to_dest
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = candidate
    
    return int(best_node)
# EVOLVE-BLOCK-END
