# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a 2-step lookahead regret-minimization framework.
    
    Design:
    - Evaluates each feasible candidate by the sum of the immediate edge cost and
      the minimum cost of the subsequent edge (or return to depot if capacity is exhausted).
    - Scores candidates by negative deviation from the mean 2-step cost, steering
      away from choices that lead to expensive or infeasible continuations.
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    
    if len(unvisited) == 0:
        return 0
    
    # Filter to feasible candidates (demand fits in rest_capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    n_candidates = len(feasible_nodes)
    
    # Pre-compute demands for unvisited nodes for fast access
    # We need to check feasibility of the NEXT node for each candidate
    
    # 1. Calculate immediate costs
    dist_to_candidates = distance_matrix[current_node, feasible_nodes]
    
    # 2. Calculate 2-step costs
    # For each candidate i, find the min cost to go to any other feasible node j
    # such that demand[j] <= rest_capacity - demand[i]
    
    two_step_costs = np.zeros(n_candidates)
    
    # Get demands for feasible nodes
    feasible_demands = demands[feasible_nodes]
    
    # Get distance matrix rows for feasible nodes
    # dist_matrix_feasible[i, j] is distance from feasible_nodes[i] to feasible_nodes[j]
    dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # We also need distance from feasible nodes to depot in case we return
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    for idx in range(n_candidates):
        # Immediate cost
        c1 = dist_to_candidates[idx]
        
        # Remaining capacity after picking this node
        rem_cap = rest_capacity - feasible_demands[idx]
        
        # Find next feasible nodes among unvisited (excluding current candidate)
        # A node j is a valid next step if:
        # 1. j is in unvisited
        # 2. j != feasible_nodes[idx]
        # 3. demands[j] <= rem_cap
        
        # Get indices of all unvisited nodes that fit in rem_cap
        next_feasible_mask = demands[unvisited] <= rem_cap
        next_feasible_indices = np.where(next_feasible_mask)[0]
        
        # Exclude the current candidate's index in the unvisited array
        # We need to map feasible_nodes[idx] back to its index in unvisited
        # Or simpler: just check all unvisited nodes
        
        # Let's iterate through potential next nodes efficiently
        # We have the full unvisited list.
        # For each j in unvisited:
        # if j == feasible_nodes[idx], skip
        # if demands[j] > rem_cap, skip
        # cost = distance_matrix[feasible_nodes[idx], j]
        
        # Vectorized approach:
        # Create a mask for valid next nodes in the full unvisited array
        valid_next_mask = next_feasible_mask.copy()
        
        # We need to exclude the current node.
        # Find the index of feasible_nodes[idx] in unvisited
        # np.where(unvisited == feasible_nodes[idx])[0]
        current_idx_in_unvisited = np.where(unvisited == feasible_nodes[idx])[0][0]
        valid_next_mask[current_idx_in_unvisited] = False
        
        if np.any(valid_next_mask):
            # Get distances from current candidate to all valid next nodes
            # distance_matrix[feasible_nodes[idx], unvisited[valid_next_mask]]
            next_costs = distance_matrix[feasible_nodes[idx], unvisited[valid_next_mask]]
            min_next_cost = np.min(next_costs)
        else:
            # If no next customer is feasible, we must return to depot
            min_next_cost = dist_to_depot[idx]
            
        two_step_costs[idx] = c1 + min_next_cost
    
    # 3. Calculate Regret Scores
    mean_two_step_cost = np.mean(two_step_costs)
    
    # Regret: negative of (cost - mean_cost)
    # We want to minimize cost, so we want high regret score for low cost.
    # Score = -(cost - mean) = mean - cost
    regret_scores = mean_two_step_cost - two_step_costs
    
    # 4. Tie-breaker / Capacity Awareness
    # If costs are very close, prefer nodes with lower demand to preserve capacity?
    # Or prefer nodes that are closer to the centroid?
    # Let's add a small penalty for high demand if capacity is tight.
    # Tightness = 1 - (rem_cap / total_remaining_demand)
    
    total_unvisited_demand = np.sum(demands[unvisited])
    
    if total_unvisited_demand > 0:
        # Capacity tightness factor
        # If rem_cap is small relative to remaining demand, high demand nodes are risky
        # But the 2-step lookahead already handles this to some extent (if rem_cap is low,
        # few next nodes are feasible, potentially forcing depot return which is costly).
        
        # Let's add a slight preference for nodes that are "central" or have lower demand
        # to avoid getting stuck.
        # Normalized demand penalty
        max_demand = np.max(demands[unvisited])
        if max_demand > 0:
             demand_penalty = feasible_demands / max_demand
        else:
             demand_penalty = np.zeros(n_candidates)
        
        # Combine: 95% regret, 5% demand penalty (lower demand is better)
        # We want to subtract penalty, so we subtract a positive value for high demand.
        combined_scores = regret_scores - 0.05 * demand_penalty
    else:
        combined_scores = regret_scores

    best_idx = np.argmax(combined_scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
