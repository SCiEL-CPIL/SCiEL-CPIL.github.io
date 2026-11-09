"""Combined program: structural Steiner/potential-field idea + refinement machinery."""
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
    # Safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def _potential_field_score(feasible_arr, current_node, depot, dm, demands, rest_capacity):
    """
    Compute a Steiner-tree-inspired potential field score for each feasible node.
    
    The idea: model unvisited customers as a spatial point cloud. The cost of visiting
    a node is approximated by how well it fits into the "convex hull" of feasible points.
    Nodes on the perimeter of dense clusters get lower scores, encouraging the heuristic
    to trace cluster boundaries rather than jumping to isolated nodes.
    
    For each candidate c:
    1. perimeter_potential: how central c is among feasible nodes (lower = more peripheral)
       We compute the weighted mean distance from c to all other feasible nodes.
       Lower mean distance = more central. We want to REWARD peripheral nodes,
       so we use (max_central - central) as a bonus.
    2. cluster_density: how many feasible nodes are "close" to c (within a radius)
       Higher density = c is in a dense cluster = good to visit now.
    3. immediate_dist: distance from current_node to c.
    4. depot_proximity: distance from c to depot (penalize if far, to keep routes compact).
    5. savings: classic savings score to capture merge potential.
    
    Combined score (minimize):
    score = immediate_dist 
            - w1 * norm_peripheral_bonus   (prefer peripheral/cluster-edge nodes)
            + w2 * norm_density            (prefer high-density areas)
            + w3 * norm_depot_dist         (prefer nodes not too far from depot)
            - w4 * norm_savings            (prefer high savings)
    """
    n = len(feasible_arr)
    if n == 1:
        return np.array([0.0])
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)].copy()
    
    # 1. Centroid/peripheral potential
    # For each node c, compute weighted mean distance to all other feasible nodes
    # Higher weight = heavier demand, so heavy nodes anchor the "center"
    demands_arr = np.array([demands[i] for i in feasible_arr], dtype=np.float64)
    total_demand = demands_arr.sum()
    if total_demand < 1e-9:
        weights = np.ones(n) / n
    else:
        weights = demands_arr / total_demand
    
    # central_score[c] = sum_f (w_f * dm[c, f])
    # Lower central_score = more central
    # We want to reward PERIPHERAL nodes, so peripheral_bonus = max_central - central_score
    central_scores = sub_dm @ weights  # shape (n,)
    
    c_min = np.min(central_scores)
    c_max = np.max(central_scores)
    c_range = c_max - c_min
    if c_range > 1e-9:
        # Normalize to [0, 1]: 0 = most central, 1 = most peripheral
        norm_peripheral = (central_scores - c_min) / c_range
    else:
        norm_peripheral = np.zeros(n)
    
    # 2. Cluster density: count of feasible nodes within a radius of each candidate
    # Use a dynamic radius based on the median pairwise distance
    flat_dm = sub_dm[~np.eye(n, dtype=bool)]
    if len(flat_dm) > 0:
        median_dist = np.median(flat_dm)
        radius = max(median_dist * 1.5, 1e-9)
    else:
        radius = 1.0
    
    # density[c] = number of feasible nodes f (f != c) with dm[c, f] <= radius
    density = (sub_dm <= radius).sum(axis=1) - 1  # exclude self
    max_density = np.max(density)
    if max_density > 0:
        norm_density = density / max_density
    else:
        norm_density = np.zeros(n)
    
    # 3. Immediate distance from current node
    dists_to_current = dm[current_node, feasible_arr]
    
    # 4. Distance to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # 5. Savings: savings(i) = d(current, depot) + d(depot, i) - d(current, i)
    # This captures how much we save by visiting i now vs. separately
    savings = dm[current_node, depot] + dists_to_depot - dists_to_current
    max_savings = np.max(savings)
    if max_savings > 1e-9:
        norm_savings = savings / max_savings
    else:
        norm_savings = np.zeros(n)
    
    # Normalize distances for comparable scaling
    d_cur_max = np.max(dists_to_current)
    if d_cur_max > 1e-9:
        norm_cur = dists_to_current / d_cur_max
    else:
        norm_cur = np.zeros(n)
    
    d_dep_max = np.max(dists_to_depot)
    if d_dep_max > 1e-9:
        norm_dep = dists_to_depot / d_dep_max
    else:
        norm_dep = np.zeros(n)
    
    # Weights: tuned to balance the different signals
    w_peripheral = 0.4    # Reward being on the edge of clusters
    w_density = 0.3       # Reward being in dense areas
    w_depot = 0.2         # Penalize being far from depot
    w_savings = 0.5       # Reward high savings
    
    # Composite score (minimize)
    score = (norm_cur 
             - w_peripheral * norm_peripheral 
             + w_density * norm_density 
             + w_depot * norm_dep 
             - w_savings * norm_savings)
    
    return score

def _greedy_complete_route(start_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Fast greedy heuristic to complete a route from start_node using savings-based scoring.
    Returns the total distance of the completed route (from start_node to depot).
    """
    total_dist = 0.0
    current = start_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            feas_arr = np.array(feasible)
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            savings = dm[current, depot] + dists_to_depot - dists_to_current
            scores = dists_to_current - 0.35 * savings + 0.1 * dists_to_depot
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
        
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
    
    total_dist += dm[current, depot]
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Steiner-tree-inspired potential field with savings-based refinement.
    Models unvisited customers as a spatial point cloud and scores each candidate
    by how well it fits into the perimeter of dense clusters, combined with
    classic savings and proximity signals.
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
    
    # Use potential field scoring for larger sets, fallback to savings for small sets
    if n <= 6:
        # For very small sets, use exhaustive 1-step lookahead
        best_cost = float('inf')
        best_node = feasible[0]
        for cand in feasible:
            sim_remaining = [x for x in unvisited if x != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            cost = dm[current_node, cand] + _greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            if cost < best_cost:
                best_cost = cost
                best_node = cand
        return int(best_node)
    else:
        # Use potential field scoring
        scores = _potential_field_score(feasible_arr, current_node, depot, dm, demands, rest_capacity)
        best_idx = np.argmin(scores)
        return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
