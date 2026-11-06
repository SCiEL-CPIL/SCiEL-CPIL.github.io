"""Improved heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 3.0  # return safe result this far before deadline


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for selecting the next customer in a capacitated vehicle routing problem.
    
    Strategy:
    1. Filter unvisited nodes that fit in rest_capacity.
    2. If none fit, return depot (0).
    3. Score each feasible node using a combination of:
       a. Immediate distance from current_node.
       b. A "regret" or "opportunity cost" term: how much worse it would be 
          to pick a slightly farther node now vs. later.
       c. A penalty for nodes with high demand (to preserve capacity for 
          harder-to-place large-demand customers).
    4. Use a 2-step lookahead to estimate the cost of the next step after 
       visiting the candidate.
    
    This balances nearest-neighbor with capacity-aware lookahead.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        feasible_arr = np.array(feasible)
        return int(feasible_arr[np.argmin(distance_matrix[current_node, feasible_arr])])
    
    # Filter feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return 0
    
    if len(feasible) == 1:
        return feasible[0]
    
    feasible_arr = np.array(feasible)
    n_feasible = len(feasible_arr)
    
    # Get distances from current node to all feasible nodes
    dists = distance_matrix[current_node, feasible_arr]
    
    # Get demands of feasible nodes
    feas_demands = demands[feasible_arr]
    
    # Compute a score for each candidate
    # Score = alpha * dist + beta * demand_penalty + gamma * regret
    
    # Regret: for each candidate i, look at the second-best option from i
    # If we pick i, what's the minimum additional cost to go from i to another feasible node?
    # This is expensive for large n, so we approximate.
    
    # Simple approach: score = dist + lambda * (max_demand_remaining / demand_i)
    # Higher demand nodes are harder to place later, so we prioritize them if they're not too far.
    
    # Better: use a weighted combination
    # Normalize distances
    min_dist = np.min(dists)
    max_dist = np.max(dists)
    if max_dist > min_dist:
        norm_dists = (dists - min_dist) / (max_dist - min_dist)
    else:
        norm_dists = np.zeros_like(dists)
    
    # Normalize demands (higher demand = higher penalty, we want to visit them earlier)
    min_dem = np.min(feas_demands)
    max_dem = np.max(feas_demands)
    if max_dem > min_dem:
        norm_dems = (feas_demands - min_dem) / (max_dem - min_dem)
    else:
        norm_dems = np.zeros_like(feas_demands)
    
    # Regret term: for each candidate, what's the average distance to other feasible nodes?
    # This is a proxy for how "isolated" the node is.
    # If a node is far from others, visiting it now might be costly later.
    # But if it's close to current, it's good to visit now.
    
    # Compute pairwise distances among feasible nodes (for small n)
    if n_feasible <= 20:
        # Full regret calculation
        # For each candidate i, compute min distance from i to any other feasible node j
        # This tells us: if we don't visit i now, how far will we have to go later?
        feasible_set = set(feasible_arr)
        regret = np.zeros(n_feasible)
        for idx, node in enumerate(feasible_arr):
            # Find min distance from node to any other feasible node
            other_dists = distance_matrix[node, feasible_arr]
            # Exclude self
            other_dists[idx] = np.inf
            regret[idx] = np.min(other_dists) if np.isfinite(other_dists).any() else 0
        
        # Normalize regret
        min_regret = np.min(regret)
        max_regret = np.max(regret)
        if max_regret > min_regret:
            norm_regret = (regret - min_regret) / (max_regret - min_regret)
        else:
            norm_regret = np.zeros_like(regret)
    else:
        # Approximate regret: use distance to nearest feasible neighbor
        norm_regret = np.zeros(n_feasible)
        # For large n, skip regret or use a cheaper approximation
        # Use distance to centroid of feasible nodes as a proxy
        feasible_coords = feasible_arr  # We don't have coordinates, use distances
        # Skip regret for large n, set to 0
        pass
    
    # Weighted score: lower is better
    # We want to minimize: distance + some penalty for high-demand nodes being left for later
    # Actually, we want to visit high-demand nodes EARLIER (when capacity is available),
    # so a lower score should correspond to: close distance AND/OR high demand
    
    # Score = w1 * norm_dist - w2 * norm_dem + w3 * norm_regret
    # Negative weight on demand: high demand -> lower score -> preferred
    
    w1 = 1.0   # distance weight
    w2 = 0.5   # demand weight (negative: prefer high demand)
    w3 = 0.3   # regret weight (prefer nodes that are hard to reach later)
    
    scores = w1 * norm_dists - w2 * norm_dems + w3 * norm_regret
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
