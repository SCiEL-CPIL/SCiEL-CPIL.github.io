"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 10s margin before 120s limit


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next customer to visit using a combined heuristic:
    1. Clarke-Wright savings (primary signal)
    2. Capacity-aware lookahead penalty (avoid getting stuck)
    3. Geometric sweep angle (for smooth routes)
    """
    if time.time() > DEADLINE:
        return 0

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    if len(unvisited) == 0:
        return 0

    # Filter by capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Get distance matrix as numpy array for vectorized ops
    dm = np.asarray(distance_matrix)
    
    # Calculate distances
    d_current = dm[current_node, feasible_nodes]
    d_depot = dm[depot, feasible_nodes]
    
    # Clarke-Wright savings: prefer nodes where d(current, node) is small relative to d(depot, node)
    # Savings = d(depot, current) + d(depot, node) - d(current, node)
    # But we're extending a route, so the relevant saving is:
    # If we go current -> node -> depot, cost is d(current,node) + d(node,depot)
    # If we go current -> depot, cost is d(current,depot)
    # The "regret" of not visiting node is related to how much we save by visiting it now
    # vs going back to depot.
    
    # Simpler: use the standard savings heuristic adapted for route extension
    # Score should favor:
    # 1. Short edges from current
    # 2. Nodes that are "close" to the depot (to make it easy to return later)
    # 3. Avoid nodes that would leave us with too little capacity for remaining nodes
    
    # Base score: minimize d_current (go to nearest feasible node)
    # But also consider the "cost to close" after visiting
    # After visiting node i, we'll need to go to depot or another node
    # Approximate future cost as d_depot[i] (distance from node to depot)
    
    # Combined cost: d_current[i] + d_depot[i]
    # This favors nodes that are close to both current and depot
    
    # But we also want to maximize savings: d_depot[current] + d_depot[i] - d_current[i]
    # Minimizing this is equivalent to minimizing d_current[i] - d_depot[i]
    
    # Let's use a weighted combination:
    # score = alpha * d_current + beta * d_depot + gamma * capacity_penalty
    
    # Capacity-aware penalty: if visiting node i leaves us with very little capacity
    # relative to the minimum demand of remaining nodes, penalize it
    remaining_demands = np.array([demands[node] for node in feasible_nodes])
    min_remaining_demand = np.min(remaining_demands)
    
    # After visiting node i, remaining capacity = rest_capacity - demands[i]
    # If this is less than some threshold relative to min demand, add penalty
    remaining_cap_after = rest_capacity - remaining_demands
    
    # Penalty if remaining capacity is tight
    # Use a soft penalty: max(0, min_demand - remaining_cap_after) * scale
    capacity_penalty = np.maximum(0, min_remaining_demand - remaining_cap_after) * 0.5
    
    # Geometric sweep: prefer nodes that are in the same angular sector as the previous node
    # This helps create smooth routes
    # Calculate angles from depot
    angles = np.arctan2(dm[feasible_nodes, 1] - dm[depot, 1], dm[feasible_nodes, 0] - dm[depot, 0])
    
    # Angle of current node
    angle_current = np.arctan2(dm[current_node, 1] - dm[depot, 1], dm[current_node, 0] - dm[depot, 0])
    
    # Angular difference (circular)
    angle_diff = np.abs(angles - angle_current)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Normalize angular difference to [0, 1]
    angle_diff_norm = angle_diff / np.pi
    
    # Combine scores
    # Primary: Clarke-Wright style savings
    cw_score = d_current - d_depot  # minimize this
    
    # Secondary: total path cost approximation
    path_cost = d_current + d_depot
    
    # Tertiary: capacity penalty
    # Quaternary: angular smoothness
    
    # Weighted sum (tuned for CVRP)
    scores = 1.0 * cw_score + 0.3 * path_cost + 1.0 * capacity_penalty + 0.2 * angle_diff_norm
    
    # Find the best node
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
