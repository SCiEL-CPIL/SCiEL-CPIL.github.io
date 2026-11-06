"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 110.0


def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE


def _soft_min(a, beta=5.0):
    """Smooth min via log-sum-exp trick: -1/beta * log(sum(exp(-beta*a)))"""
    a = np.asarray(a, dtype=float)
    a_min = np.min(a, axis=1, keepdims=True)
    # Numerical stability
    exp_terms = np.exp(-beta * (a - a_min))
    return a_min.squeeze(axis=1) - (1.0 / beta) * np.log(np.sum(exp_terms, axis=1))


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited[feasible_mask]
    
    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Time guard: if we're close to deadline, fall back to simple nearest-neighbor
    if _time_exceeded():
        dist_to_feasible = distance_matrix[current_node, feasible_nodes]
        best_idx = np.argmin(dist_to_feasible)
        return int(feasible_nodes[best_idx])
    
    # Compute distances from current node to feasible nodes
    dist_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible > 1:
        # For each candidate i, estimate the cost of going to the next best feasible node after it
        # Using soft-min (log-sum-exp) for a smooth surrogate of the min operation
        # This avoids the nonsmoothness of hard min and provides gradient-like information
        
        # dist_matrix between feasible nodes
        feas_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        
        # Set diagonal to a large value (not inf, to avoid numerical issues with exp)
        diag_idx = np.diag_indices(n_feasible)
        feas_dist[diag_idx] = np.max(feas_dist) * 1.1 + 1.0
        
        # Soft-min: smooth approximation of min distance to any other feasible node
        # Higher beta -> closer to hard min; lower beta -> smoother
        beta = 4.0
        soft_min_next = _soft_min(feas_dist, beta=beta)
        
        # Also consider a 3-step lookahead for better route planning
        # For each pair (i, j), find the soft-min distance from j to any k (k != i, j)
        # This is more expensive but gives better lookahead
        if n_feasible <= 30:
            # 3-step lookahead: for each i, find j that minimizes dist(i,j) + softmin(j, others except i)
            alpha2 = 0.15
            # For each j, compute soft-min to all others except j itself
            # We already have soft_min_next[j] which is min over all k != j
            # For 3-step: score_i = dist(c,i) + alpha * min_j [dist(i,j) + alpha2 * softmin(j, k!=i,j)]
            # Simplified: use the 2-step soft-min as primary, add small 3-step correction
            
            # For each i, the best 2-step path cost
            # path_cost[i] = min_j (feas_dist[i,j] + alpha2 * soft_min_to_others[j, excluding i])
            # Approximate: soft_min_next[j] already excludes j, but may include i
            # Good enough as a heuristic correction
            two_step = feas_dist + alpha2 * soft_min_next[np.newaxis, :]
            best_two_step = np.min(two_step, axis=1)
            
            # Combine: immediate + 2-step lookahead
            alpha = 0.25
            scores = dist_to_feasible + alpha * best_two_step
        else:
            # For larger sets, just use 2-step soft-min
            alpha = 0.3
            scores = dist_to_feasible + alpha * soft_min_next
    else:
        scores = dist_to_feasible
    
    # Capacity-aware penalty: if rest_capacity is very low relative to remaining demands,
    # prefer nodes with lower demand to keep options open
    if rest_capacity < 15:
        feas_demands = demands[feasible_nodes].astype(float)
        # Penalize high-demand nodes when capacity is tight
        cap_penalty = 0.1 * (feas_demands / max(rest_capacity, 1))
        scores = scores + cap_penalty
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
