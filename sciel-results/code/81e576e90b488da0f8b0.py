# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    k = len(unvisited_arr)

    # Feasibility: demand must fit in remaining capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    m = len(feas)

    # If only one feasible, take it
    if m == 1:
        return int(feas[0])

    # Distances from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    # Dynamic Regret-Minimization Strategy:
    # We want to avoid "outlier-long" edges that might lead to inefficient routing or
    # premature capacity issues due to poor clustering.
    # 
    # Score_j = d(current, j) - beta * (Average distance from j to other feasible nodes)
    #
    # The term (Average distance from j to other feasible nodes) acts as a proxy for 
    # how "central" or "connected" node j is to the rest of the feasible cluster.
    # A node with low average distance to others is well-connected (low regret for 
    # choosing it in terms of future connectivity). 
    # A node with high average distance is an outlier (high regret).
    #
    # By subtracting this average distance (scaled by beta), we effectively:
    # 1. Reward nodes that are close to current (d_cur).
    # 2. Reward nodes that are close to other feasible nodes (compactness/centralization).
    # 3. Penalize nodes that are far from other feasible nodes (outliers).
    
    # Compute average distance from each feasible node to all other feasible nodes
    # We use the full unvisited set for the "regret" calculation to consider the whole
    # remaining workload, but weighted by feasibility? 
    # Actually, the prompt says "average edge cost of the remaining feasible set".
    # So we compute distances between feasible nodes only.
    
    # Get the submatrix of distance_matrix for feasible nodes
    # d_feas_feas[i, j] = distance between feas[i] and feas[j]
    d_feas_feas = distance_matrix[np.ix_(feas, feas)]
    
    # Sum of distances from each feasible node to all other feasible nodes
    # Exclude self (diagonal is 0 anyway)
    row_sums = d_feas_feas.sum(axis=1)
    
    # Average distance to other feasible nodes
    # If m > 1, divide by m-1. If m==1, we already returned.
    avg_d_to_feas = row_sums / (m - 1)
    
    # Beta controls the weight of the regret/compactness term
    # A higher beta makes the heuristic more "cluster-aware" and less "greedy-nearest".
    # Current program used alpha=0.4 for a similar but slightly different term 
    # (mean to ALL unvisited, not just feasible).
    # Let's try beta = 0.5 to strongly favor compact clusters of feasible nodes.
    beta = 0.5
    
    scores = d_cur - beta * avg_d_to_feas
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(feas[best_idx])
# EVOLVE-BLOCK-END
