"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return time.time() >= _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour construction using a multi-criteria heuristic.
    
    The heuristic combines:
    1. Immediate distance from current to candidate (nearest neighbor component)
    2. Distance from candidate to destination (return-leg component)
    3. A soft 2-step lookahead using log-sum-exp for smoothness
    
    This biases selection toward nodes that are close to the current node
    AND have good prospects for returning to the destination.
    """
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if close to limit, fall back to fast nearest-neighbor
    if _deadline_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current node to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited nodes to destination
    d_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For 2-step lookahead: for each candidate i, estimate min distance from i to any other unvisited j
    # Then the 2-step cost for candidate i is: d(current,i) + min_j(d(i,j)) + d(j,dest) approximation
    # To keep it efficient, we use a soft-min approximation
    
    # Normalize distances to [0, 1] range for stable computation
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-10)
    d_to_dest_norm = d_to_dest / (np.max(d_to_dest) + 1e-10)
    
    # 2-step lookahead: for each candidate i, compute a soft estimate of the best next hop
    # This is O(n^2) but n <= 50 so it's fast
    n = len(unvisited_nodes)
    
    if n > 1:
        # Distance matrix among unvisited nodes
        d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # For each candidate i, compute soft-min of (d(i,j) + d(j,dest)) over j != i
        # Using log-sum-exp for smoothness
        # Cost matrix: cost[i,j] = d(i,j) + d(j,dest)
        cost_2step = d_unvisited + d_to_dest[np.newaxis, :]
        
        # Mask self-transitions
        np.fill_diagonal(cost_2step, np.inf)
        
        # Soft-min via log-sum-exp with temperature
        # soft_min_i = -log(sum_j exp(-cost[i,j] / T)) * T
        # We want to minimize, so use negative log-sum-exp
        T = 0.1  # Temperature for soft-min
        neg_cost = -cost_2step / T
        # Subtract max for numerical stability
        max_neg = np.max(neg_cost, axis=1, keepdims=True)
        # Handle inf values
        max_neg = np.where(np.isinf(max_neg), 0, max_neg)
        exp_neg = np.exp(neg_cost - max_neg)
        # Replace inf in original with 0 in exp (since exp(-inf)=0)
        exp_neg = np.where(np.isinf(cost_2step), 0, exp_neg)
        sum_exp = np.sum(exp_neg, axis=1)
        sum_exp = np.maximum(sum_exp, 1e-300)
        soft_min_2step = -T * (np.log(sum_exp) + max_neg[:, 0])
        
        # Normalize soft_min_2step
        soft_min_2step_norm = soft_min_2step / (np.max(soft_min_2step) + 1e-10)
    else:
        soft_min_2step_norm = np.zeros(n)
    
    # Combine criteria with weights
    # Weight immediate distance, destination distance, and 2-step lookahead
    w1 = 0.4  # Weight for immediate distance
    w2 = 0.3  # Weight for destination distance  
    w3 = 0.3  # Weight for 2-step lookahead
    
    score = w1 * d_from_current_norm + w2 * d_to_dest_norm + w3 * soft_min_2step_norm
    
    # Select the node with minimum combined score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
