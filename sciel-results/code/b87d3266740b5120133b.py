"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Direct savings-based insertion heuristic with lookahead.
    Uses a composite score that combines:
    1. Immediate distance from current node
    2. Savings potential (how well this node connects to others and depot)
    3. Demand-weighted position (heavier nodes earlier if close)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: fall back to nearest feasible
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
    
    feasible_arr = np.array(feasible, dtype=np.int64)
    n = len(feasible_arr)
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    if n > 1:
        # For each candidate i, compute:
        # 1. min_dist_to_other: minimum distance from i to any other feasible node
        #    (penalize isolated nodes)
        # 2. savings_score: how much savings this node provides with the rest
        #    savings(i,j) = d(depot,i) + d(depot,j) - d(i,j)
        #    We sum savings with all others to get total merge potential
        
        # Min distance to any other feasible node (excluding self)
        np.fill_diagonal(sub_dm, np.inf)
        min_dist_to_other = np.min(sub_dm, axis=1)
        
        # Restore diagonal for savings computation
        np.fill_diagonal(sub_dm, 0)
        
        # Savings matrix: S[i,j] = d_depot[i] + d_depot[j] - d(i,j)
        d_depot = dists_to_depot
        savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
        
        # Total savings for each node with all others
        total_savings = np.sum(savings_matrix, axis=1)
        
        # Normalize
        max_savings = np.max(total_savings) if np.max(total_savings) > 0 else 1.0
        norm_savings = total_savings / max_savings
        
        # Normalize min_dist_to_other
        max_min_dist = np.max(min_dist_to_other) if np.max(min_dist_to_other) > 0 else 1.0
        norm_min_dist = min_dist_to_other / max_min_dist
    else:
        norm_savings = np.array([0.0])
        norm_min_dist = np.array([0.0])
    
    # Demand normalization
    demands_arr = np.array([demands[i] for i in feasible], dtype=np.float64)
    max_demand = np.max(demands_arr) if np.max(demands_arr) > 0 else 1.0
    norm_demand = demands_arr / max_demand
    
    # Composite score with 2-step lookahead:
    # For each candidate i, estimate the marginal cost of inserting i into the route.
    # The route so far ends at current_node. After visiting i, we'll visit some j,
    # then eventually return to depot.
    #
    # Marginal cost of picking i:
    #   d(current, i) + [estimated future cost from i]
    #
    # Estimated future cost from i:
    #   The best 2nd node j to visit after i would minimize d(i,j) + d(j,depot)
    #   among remaining feasible nodes. But we also need to account for the fact
    #   that other nodes still need to be visited.
    #
    # A better surrogate: for candidate i, the "effective cost" is:
    #   d(current, i) + min_j [d(i,j) + d(j, depot)]  for j != i, j feasible
    #   If no other node, just d(current, i) + d(i, depot)
    #
    # This is a smooth surrogate for the true remaining route cost.
    # We also add a savings bonus: nodes that have high total savings with others
    # are good to pick early because they "bridge" multiple clusters.
    
    if n > 1:
        # For each candidate i, find the best 2nd node j (j != i)
        # that minimizes d(i,j) + d(j, depot)
        # cost_ij_jd[i, j] = d(i,j) + d(j, depot)
        cost_ij_jd = sub_dm + dists_to_depot[None, :]
        # Exclude self (set diagonal to inf)
        np.fill_diagonal(cost_ij_jd, np.inf)
        # For each i, the best 2nd step cost
        best_second_step = np.min(cost_ij_jd, axis=1)
        
        # Total estimated cost for candidate i:
        # d(current, i) + best_second_step[i]
        # But we also need to eventually return to depot from the last node.
        # A rough correction: subtract d(i, depot) from the estimate since
        # the actual route will end at depot, not at the 2nd node.
        # Actually, let's keep it simple: the 2-step lookahead already
        # accounts for going i -> j -> (eventually depot).
        
        # Normalize best_second_step
        max_bss = np.max(best_second_step) if np.max(best_second_step) > 0 else 1.0
        norm_bss = best_second_step / max_bss
        
        # Normalize dists_to_current
        max_dtc = np.max(dists_to_current) if np.max(dists_to_current) > 0 else 1.0
        norm_dtc = dists_to_current / max_dtc
        
        # Savings component (already normalized)
        # Use a weighted combination:
        # score = norm_dtc + gamma * norm_bss - delta * norm_savings
        # Higher norm_savings is better (reduces total route length)
        
        gamma = 0.5   # Weight for 2-step lookahead cost
        delta = 0.3   # Weight for savings bonus
        
        scores = norm_dtc + gamma * norm_bss - delta * norm_savings
    else:
        scores = dists_to_current
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
