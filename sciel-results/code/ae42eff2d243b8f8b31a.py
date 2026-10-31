"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 140.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Soft-Min Lookahead heuristic.
    
    Strategy:
    1. Check time budget.
    2. For each candidate node `c` in `unvisited_nodes`:
       a. Immediate cost: dist(current, c)
       b. Future cost estimate: Soft-min of dist(c, c') for all c' in unvisited \ {c}.
          This smooths the "nearest neighbor" choice for the next step.
       c. Destination bias: dist(c, destination) to prevent drifting away.
    3. Select the candidate minimizing the weighted sum.
    """
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: Nearest Neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Extract sub-matrix of distances among unvisited nodes
    # dist_sub[i, j] = distance between unvisited_nodes[i] and unvisited_nodes[j]
    dist_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Immediate costs from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Destination costs from all unvisited nodes to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate Soft-Min future cost for each candidate c
    # We want softmin_{c' != c} dist(c, c')
    # softmin(x) = -1/tau * log(sum(exp(-x/tau)))
    
    # To handle the "exclude self" part:
    # Set diagonal to a large value (inf) before summing, so exp(-inf/tau) = 0.
    # This effectively removes the self-term from the sum.
    mask = np.eye(n_unvisited, dtype=bool)
    dist_sub_masked = dist_sub.copy()
    dist_sub_masked[mask] = np.inf
    
    # Scale for soft-min. 
    # Distances are typically ~1.0. tau=0.5 is a good balance between sharp (greedy) and smooth (global).
    tau = 0.5
    # Avoid overflow in exp: x/tau can be large negative. 
    # exp(-inf) is 0. exp(-large) is 0.
    # We compute sum(exp(-dist/tau)).
    # To improve numerical stability, we can subtract the min, but here we just want the sum.
    # Since dist >= 0, -dist/tau <= 0. exp is in (0, 1]. No overflow.
    
    exp_terms = np.exp(-dist_sub_masked / tau)
    sum_exp = np.sum(exp_terms, axis=1)
    
    # softmin = -1/tau * log(sum_exp)
    # If sum_exp is 0 (shouldn't happen for n>1), handle it.
    with np.errstate(divide='ignore', invalid='ignore'):
        soft_min_costs = -np.log(sum_exp) / tau
    
    # Replace any inf/nan with a large penalty
    soft_min_costs = np.where(np.isfinite(soft_min_costs), soft_min_costs, 1e6)
    
    # Combine costs
    # Weights: 
    # Immediate cost is primary.
    # Future cost is secondary but important for lookahead.
    # Destination bias is tertiary to ensure we return to destination.
    
    w_immediate = 1.0
    w_future = 0.5
    w_dest = 0.2
    
    scores = w_immediate * dists_from_current + w_future * soft_min_costs + w_dest * dists_to_dest
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
