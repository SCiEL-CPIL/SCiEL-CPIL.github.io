"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Lookahead insertion heuristic with savings and continuation cost.
    For each feasible candidate, estimate the marginal cost of adding it to the route.
    We approximate the future route cost by considering:
    1. Distance from current to candidate (immediate cost)
    2. A penalty for how "expensive" it will be to connect the candidate to the rest
       of the route or depot, weighted by the remaining demand.
    
    This differs from pure nearest-neighbor by penalizing nodes that are isolated
    or far from the centroid of remaining feasible nodes, encouraging compact routes.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Compute the sub-distance matrix among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    np.fill_diagonal(sub_dm, 0)
    
    # Average distance to other feasible nodes
    avg_dist_to_others = np.sum(sub_dm, axis=1) / (n - 1)
    
    # Compute savings matrix among feasible nodes
    # S[i,j] = d(depot, i) + d(depot, j) - d(i, j)
    d_depot = dists_to_depot
    savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
    np.fill_diagonal(savings_matrix, -np.inf)
    
    # Use soft-max (log-sum-exp) for smooth savings estimate instead of hard max
    # This avoids the discontinuity of argmax and gives a smoother gradient
    # Soft-max: log(sum(exp(x/tau))) * tau
    tau_s = 1.0  # temperature parameter for savings
    # Shift for numerical stability
    shifted_savings = savings_matrix / tau_s
    max_shifted = np.max(shifted_savings, axis=1, keepdims=True)
    # Handle -inf values: if all are -inf, max is -inf
    exp_shifted = np.exp(shifted_savings - max_shifted)
    exp_shifted = np.where(np.isfinite(exp_shifted), exp_shifted, 0)
    sum_exp = np.sum(exp_shifted, axis=1)
    # Soft-max value (in original scale)
    soft_max_savings = tau_s * (max_shifted.squeeze() + np.log(sum_exp + 1e-10))
    # Where sum_exp was 0 (all -inf), set to 0
    soft_max_savings = np.where(sum_exp > 0, soft_max_savings, 0)
    
    # Continuation cost: after visiting candidate i, what's the min distance to any other feasible
    # node or to depot? Use soft-min for smoothness
    min_sub = np.min(sub_dm, axis=1)
    # Combine with depot distance using soft-min
    # soft-min(a, b) = -soft-max(-a, -b)
    tau_c = 1.0
    combined_cont = np.stack([-min_sub, -dists_to_depot], axis=1)
    max_combined = np.max(combined_cont, axis=1, keepdims=True)
    exp_combined = np.exp(combined_cont - max_combined)
    sum_exp_c = np.sum(exp_combined, axis=1)
    soft_max_combined = tau_c * (max_combined.squeeze() + np.log(sum_exp_c + 1e-10))
    soft_min_cont = -soft_max_combined
    
    # Normalize savings and continuation cost to be comparable to distances
    max_s = np.max(np.abs(soft_max_savings)) if np.max(np.abs(soft_max_savings)) > 0 else 1.0
    norm_savings = soft_max_savings / max_s
    
    max_c = np.max(soft_min_cont) if np.max(soft_min_cont) > 0 else 1.0
    norm_cont = soft_min_cont / max_c
    
    # Composite score:
    # score = dist_to_current - delta * norm_savings + gamma * norm_cont
    # We want to minimize this score.
    delta = 0.5  # Weight for savings (prefer high savings)
    gamma = 0.3  # Weight for continuation cost (penalize high continuation cost)
    
    scores = dists_to_current - delta * norm_savings + gamma * norm_cont
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
