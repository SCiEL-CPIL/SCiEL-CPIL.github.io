# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a dynamic regret-minimization framework.
    
    Design:
    - Score each candidate by negative edge cost relative to the mean edge cost
      of all feasible candidates, steering away from outlier-long edges.
    - Add a capacity-aware penalty that discourages picking high-demand nodes
      when remaining capacity is tight relative to the remaining workload.
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
    dist_to_candidates = distance_matrix[current_node, feasible_nodes]
    
    # Mean edge cost over the feasible set
    mean_cost = np.mean(dist_to_candidates)
    
    # Regret score: negative of (cost - mean_cost)
    # This rewards nodes that are cheaper than average and penalizes expensive ones
    regret_scores = -(dist_to_candidates - mean_cost)
    
    # Capacity-aware penalty:
    # After picking node i, remaining capacity = rest_capacity - demand[i]
    # Remaining nodes to serve = len(unvisited) - 1
    # If remaining capacity / remaining nodes < some threshold, penalize more
    remaining_caps = rest_capacity - demands[feasible_nodes]
    num_remaining = len(unvisited) - 1
    
    if num_remaining > 0:
        # Average demand of all unvisited nodes
        avg_demand = np.mean(demands[unvisited])
        # Capacity per remaining node if we pick this candidate
        cap_per_node = remaining_caps / num_remaining
        # Deficit: how much below average demand the capacity per node is
        deficit = np.maximum(0.0, avg_demand - cap_per_node)
        # Normalize deficit by avg_demand to get a [0, 1+] penalty
        capacity_penalty = deficit / (avg_demand + 1e-9)
    else:
        capacity_penalty = np.zeros(len(feasible_nodes), dtype=float)
    
    # Combined score
    combined_scores = regret_scores - capacity_penalty
    
    best_idx = np.argmax(combined_scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
