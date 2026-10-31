"""TSP constructive heuristic: Regret-based selection with soft-min global structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using regret-based heuristic with soft-min global structure.
    
    For each candidate c, compute:
      regret(c) = second_best_connection(c) - best_connection(c)
      where connections are to other unvisited nodes + destination (with adaptive weight)
    
    Final score: d(cur, c) + alpha * regret(c) + beta * soft_min_future(c)
    
    Where:
      - regret prioritizes nodes that are "hard to fit in later"
      - soft_min_future uses a temperature-based soft min over future connections
      - alpha, beta adapt based on remaining nodes
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])

    # Adaptive parameters
    # As fewer nodes remain, destination becomes more important
    dest_weight = max(0.3, 1.0 - (n_unvisited / 40.0))
    
    # Regret weight: stronger when more nodes remain (early decisions matter more)
    alpha = 0.5 * (n_unvisited / 50.0) + 0.1
    
    # Soft-min temperature: lower temp = more greedy, higher = more exploratory
    # Use a temperature that decreases as we approach the end
    temp = 0.1 + 0.3 * (n_unvisited / 50.0)
    
    # Precompute
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For regret: compute best and second-best connections for each candidate
    # Connections include other unvisited nodes AND destination (weighted)
    # Create extended connection matrix: rows are candidates, cols are (other unvisited + dest)
    # But we can compute regret directly from submatrix + dest connections
    
    # Best and second-best to other unvisited nodes (excluding self)
    sub_with_inf_diag = submatrix.copy()
    np.fill_diagonal(sub_with_inf_diag, np.inf)
    
    # Get two smallest values per row
    sorted_sub = np.sort(sub_with_inf_diag, axis=1)
    best_conn = sorted_sub[:, 0]
    second_best_conn = sorted_sub[:, 1] if n_unvisited > 2 else sorted_sub[:, 0] + 0.001
    
    # Include destination in the connection set
    # For each candidate, its connections are: [d(c, b) for b != c] + [dest_weight * d(c, dest)]
    # Regret should consider all possible next moves from c
    
    # Compute full connection costs for each candidate
    # conn_costs[i] = [d(i, j) for j in unvisited, j != i] + [dest_weight * d(i, dest)]
    # We need top-2 smallest from this set
    
    # Combine: for each row i, we have sub_with_inf_diag[i, :] (with inf at i) and dest_weight * dists_to_dest[i]
    # Create augmented matrix
    aug_matrix = np.hstack([sub_with_inf_diag, (dest_weight * dists_to_dest)[:, np.newaxis]])
    
    # Get two smallest per row
    sorted_aug = np.sort(aug_matrix, axis=1)
    best_full = sorted_aug[:, 0]
    second_full = sorted_aug[:, 1]
    
    # Regret = second_best - best (how much worse is the alternative?)
    regrets = second_full - best_full
    
    # Soft-min future: for each candidate, compute a soft-min over future connections
    # This captures "how well-connected is this node to the rest?"
    # Use soft-min: -temp * log(sum(exp(-conn/temp)))
    # But we want a cheap approximation: use the mean of the k smallest connections
    
    # For efficiency, use the mean of the 3 smallest connections (or all if fewer)
    k = min(3, n_unvisited)
    soft_min_approx = np.mean(sorted_aug[:, :k], axis=1)
    
    # Final score: distance from current + regret penalty + soft-min bonus
    # Lower score is better
    # Regret: high regret means we should visit this node now (positive contribution to cost)
    # Soft-min: low soft-min means well-connected (negative contribution, i.e., bonus)
    scores = dists_from_current + alpha * regrets - 0.3 * soft_min_approx
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
