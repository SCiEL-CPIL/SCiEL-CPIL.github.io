"""Model-and-optimize approach for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Module-level state
_dm_cache = {}
_start_time = None
_instance_id = 0

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    global _instance_id
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
        _instance_id += 1
        _start_time = time.time()
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 140

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
            # Savings: prefer nodes that save travel by being served now vs. separately
            # saving = dm[current,depot] + dm[depot,cand] - dm[current,cand]
            # Higher saving = better; combine with proximity
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

def _improved_greedy_complete_route(start_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Improved greedy that looks ahead 1-2 steps with savings-based scoring.
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
        elif len(feasible) <= 8:
            best_cost = float('inf')
            best_node = feasible[0]
            for cand in feasible:
                sim_remaining = [n for n in remaining if n != cand]
                sim_cap = remaining_cap - demands[cand]
                if sim_cap < 0:
                    continue
                cost = dm[current, cand] + _greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
                if cost < best_cost:
                    best_cost = cost
                    best_node = cand
            next_node = best_node
        else:
            feas_arr = np.array(feasible)
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            # Savings-based scoring for consistency
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

def _density_centroid_select(current_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Density-weighted centroid selection.
    Computes a demand-weighted centroid of all feasible unvisited nodes (using
    the distance matrix as a proxy for spatial position), then scores each
    candidate by how well it maintains local spatial density. Candidates that
    are close to the centroid AND close to the current node are preferred,
    as they minimize intra-cluster travel variance.
    """
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])
    
    feas_arr = np.array(feasible)
    n_feasible = len(feas_arr)
    
    # Demand weights for density computation
    demand_weights = np.array([demands[n] for n in feas_arr], dtype=np.float64)
    total_demand = demand_weights.sum()
    if total_demand < 1e-9:
        demand_weights = np.ones_like(demand_weights)
        total_demand = demand_weights.sum()
    demand_weights = demand_weights / total_demand
    
    # Compute density-weighted centroid using distance matrix rows as "positions"
    # The centroid is a weighted average of the distance vectors to all nodes
    # This captures the spatial distribution of feasible nodes
    # centroid[i] = sum_j (w_j * dm[j, i]) for each dimension i
    # But since we don't have explicit coordinates, we use the distance to depot
    # and distance to current node as proxy features
    
    # Alternative: use the distance matrix structure directly
    # For each candidate c, compute how "central" it is among feasible nodes
    # centrality(c) = mean over all feasible nodes f of dm[c, f] weighted by demand
    
    # Compute pairwise distances among feasible nodes
    # feas_dm = dm[np.ix_(feas_arr, feas_arr)]
    
    # Distance from each feasible node to each other feasible node
    # centrality[c] = sum_f (w_f * dm[c, f]) / total_demand
    # Lower centrality = more central = better
    
    feas_idx = feas_arr
    # dm_feas = dm[np.ix_(feas_idx, feas_idx)]
    # centrality = demand_weights @ dm_feas  # shape: (n_feasible,)
    
    # For memory efficiency with large sets, compute in chunks or use the fact
    # that we only need the dot product
    # centrality[c] = sum_f (w_f * dm[c, f])
    
    # Vectorized: centrality = dm_feas @ demand_weights
    # But dm_feas could be large. Let's compute directly.
    
    # Get the submatrix
    dm_sub = dm[np.ix_(feas_idx, feas_idx)]
    centrality = dm_sub @ demand_weights  # weighted mean distance to other feasible nodes
    
    # Distance from current node to each feasible
    dist_current = dm[current_node, feas_arr]
    
    # Distance from each feasible to depot
    dist_depot = dm[feas_arr, depot]
    
    # Normalize centrality to [0, 1] for scoring
    c_min = np.min(centrality)
    c_max = np.max(centrality)
    if c_max > c_min:
        norm_central = (centrality - c_min) / (c_max - c_min)
    else:
        norm_central = np.zeros_like(centrality)
    
    # Score: prefer nodes that are:
    # 1. Close to current (dist_current)
    # 2. Central among feasible nodes (low norm_central)
    # 3. Not too far from depot (dist_depot) to keep route compact
    # Combined: minimize dist_current + alpha * norm_central * scale + beta * dist_depot
    
    # Scale centrality to be comparable to distances
    scale = c_max - c_min if c_max > c_min else 1.0
    centrality_scaled = norm_central * scale
    
    score = dist_current + 0.3 * centrality_scaled + 0.1 * dist_depot
    
    best_idx = np.argmin(score)
    return int(feas_arr[best_idx])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: For each feasible candidate, estimate the total cost of the
    remaining route if we pick that candidate. Choose the candidate that
    minimizes this estimate.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard
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
    
    n_feasible = len(feasible)
    
    # If few feasible nodes, do full lookahead
    if n_feasible <= 8:
        best_cost = float('inf')
        best_node = feasible[0]
        
        for cand in feasible:
            sim_remaining = [n for n in unvisited if n != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            sim_cost = _improved_greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            total_cost = dm[current_node, cand] + sim_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_node = cand
        
        return int(best_node)
    else:
        # Hybrid: use density-centroid scoring to filter top-k, then lookahead
        feas_arr = np.array(feasible)
        
        # Compute density-based filter score
        demand_weights = np.array([demands[n] for n in feas_arr], dtype=np.float64)
        total_demand = demand_weights.sum()
        if total_demand < 1e-9:
            demand_weights = np.ones_like(demand_weights)
            total_demand = demand_weights.sum()
        demand_weights = demand_weights / total_demand
        
        dm_sub = dm[np.ix_(feas_arr, feas_arr)]
        centrality = dm_sub @ demand_weights
        
        c_min = np.min(centrality)
        c_max = np.max(centrality)
        if c_max > c_min:
            norm_central = (centrality - c_min) / (c_max - c_min)
        else:
            norm_central = np.zeros_like(centrality)
        
        scale = c_max - c_min if c_max > c_min else 1.0
        centrality_scaled = norm_central * scale
        
        dists = dm[current_node, feas_arr]
        dists_depot = dm[feas_arr, depot]
        
        # Add savings component for consistency with route completion scoring
        savings = dm[current_node, depot] + dists_depot - dists
        score = dists + 0.3 * centrality_scaled + 0.1 * dists_depot - 0.35 * savings
        
        # Get top-k by combined score
        k = min(6, n_feasible)
        top_k_indices = np.argpartition(score, k - 1)[:k]
        top_k_nodes = [int(feas_arr[i]) for i in top_k_indices]
        
        best_cost = float('inf')
        best_node = top_k_nodes[0]
        
        for cand in top_k_nodes:
            sim_remaining = [n for n in unvisited if n != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            sim_cost = _improved_greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            total_cost = dm[current_node, cand] + sim_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_node = cand
        
        return int(best_node)
# EVOLVE-BLOCK-END
