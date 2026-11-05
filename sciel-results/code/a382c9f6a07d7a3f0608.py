# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a dynamic regret-minimization framework.
    
    For each feasible candidate (demand fits in rest_capacity), we compute:
      - edge_cost: distance from current_node to candidate
      - avg_edge_cost: mean edge cost from current_node to all remaining feasible candidates
    
    The score is: -(edge_cost - avg_edge_cost)
    We want to minimize regret, i.e., prefer edges that are below the average.
    This steers the route away from outlier-long edges.
    
    Additionally, we incorporate a capacity-aware penalty to avoid premature
    capacity exhaustion: if taking a candidate leaves very little rest_capacity
    relative to the average demand of remaining nodes, we penalize it.
    
    If no feasible customer exists, return depot (0).
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    
    if len(unvisited) == 0:
        return 0
    
    # Filter to feasible candidates (demand fits in rest_capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # Edge costs from current_node to each feasible candidate
    edge_costs = distance_matrix[current_node, feasible_nodes]
    
    # Average edge cost over all feasible candidates
    avg_edge_cost = np.mean(edge_costs)
    
    # Regret score: we want to minimize (edge_cost - avg_edge_cost),
    # i.e., prefer edges below the average. Negative of this difference
    # means we maximize (avg_edge_cost - edge_cost), so we pick the one
    # that is furthest below average (most "regret-minimizing").
    regret_scores = -(edge_costs - avg_edge_cost)
    
    # Capacity-aware penalty: if taking this candidate would leave very little
    # rest_capacity relative to what's needed for remaining nodes, penalize it.
    # Compute remaining capacity after taking each candidate
    remaining_caps = rest_capacity - demands[feasible_nodes]
    
    # Average demand of all unvisited nodes (not just feasible) as a proxy for
    # what we'll need going forward
    if len(unvisited) > 0:
        avg_demand_all = np.mean(demands[unvisited])
    else:
        avg_demand_all = 0
    
    # Penalty: if remaining_cap is low relative to avg_demand_all, add penalty
    # Scale: if remaining_cap < avg_demand_all, we're in tight capacity
    # Normalize: penalty = max(0, (avg_demand_all - remaining_cap) / max(avg_demand_all, 1))
    if avg_demand_all > 0:
        capacity_penalty = np.maximum(0, (avg_demand_all - remaining_caps) / avg_demand_all)
    else:
        capacity_penalty = np.zeros_like(remaining_caps, dtype=float)
    
    # Combined score: regret score minus capacity penalty (we want to maximize combined)
    # Since regret_scores is already "higher is better" (less regret),
    # and capacity_penalty is "higher is worse", we subtract it.
    combined_scores = regret_scores - capacity_penalty
    
    # Pick the candidate with the highest combined score
    best_idx = np.argmax(combined_scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
