"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIME_BUDGET = 140.0  # seconds, safety margin before 150s kill

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _time_remaining():
    return _TIME_BUDGET - (time.time() - _get_start_time())


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if _time_remaining() < 0.5:
        # Emergency: just nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    # Get distances from destination to all unvisited
    d_dest = distance_matrix[destination_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # 2-step lookahead: for each candidate i, estimate cost of going
    # current -> i -> best_next(i) + return consideration
    # best_next(i) is the unvisited node j != i that minimizes d(i,j)
    
    # Efficient 2-step: for each i, find min over j != i of d(i,j)
    # We can compute this by taking the two smallest distances from each i to other unvisited
    
    # d_matrix among unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i, find the two smallest values and their indices
    # Use argpartition for efficiency
    if n_unvisited > 2:
        # Get indices of two smallest in each row
        part_idx = np.argpartition(d_unvisited, 1, axis=1)[:, :2]
        # Get the two smallest values
        part_vals = np.take_along_axis(d_unvisited, part_idx, axis=1)
        
        # best_next_cost[i] = min over j != i of d(i,j)
        # Since j != i, and d(i,i)=0, we need to be careful
        # d(i,i) = 0 is always the smallest, so the second smallest is our answer
        best_next_cost = part_vals[:, 1]
        best_next_idx = part_idx[:, 1]
    else:
        # n_unvisited == 2
        best_next_cost = np.array([d_unvisited[0, 1], d_unvisited[1, 0]])
        best_next_idx = np.array([1, 0])
    
    # Score for each candidate i:
    # cost = d(current, i) + alpha * d(i, best_next(i)) + beta * d(best_next(i), destination)
    # But we don't know which node is best_next(i) for the destination distance.
    # Simplification: use d(i, dest) as proxy for the return leg
    
    # Adaptive alpha and beta based on remaining nodes
    # Early in the tour: more lookahead weight; later: less
    progress = 1.0 - n_unvisited / 50.0  # rough progress estimate
    
    alpha = 0.5 * (1.0 - progress)  # lookahead weight decreases as we finish
    beta = 0.3 * (1.0 - progress)   # destination bias decreases as we finish
    
    # Compute score for each candidate
    # score[i] = d(current, i) + alpha * best_next_cost[i] + beta * d_dest[i]
    scores = d_current + alpha * best_next_cost + beta * d_dest
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
