"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 10s margin before 120s limit


def _time_ok():
    return time.time() < DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next customer to visit using a savings-based heuristic with:
    1. Clarke-Wright savings (d_current - d_depot)
    2. Capacity-aware lookahead penalty (prefer nodes that leave room for future visits)
    3. Soft-min surrogate for closing cost to smooth the decision boundary
    """
    if not _time_ok():
        return 0

    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return 0

    # Filter by capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)

    # Calculate distances
    d_current = distance_matrix[current_node, feasible_nodes]
    d_depot = distance_matrix[depot, feasible_nodes]

    # Clarke-Wright savings: prefer nodes where going via current saves distance
    base_scores = d_current - d_depot

    # Capacity-aware lookahead: penalize nodes with high demand relative to remaining capacity
    # This helps avoid situations where we pick a high-demand node and then can't fit others
    demands_feasible = np.array([demands[node] for node in feasible_nodes])
    # Normalize demand by remaining capacity (0 to 1 range)
    demand_ratio = demands_feasible / max(rest_capacity, 1)
    # Penalty: higher when demand is large fraction of remaining capacity
    # Use a quadratic penalty to strongly discourage near-capacity picks
    capacity_penalty = 0.5 * demand_ratio ** 2

    # Soft-min surrogate for closing cost:
    # If we pick this node, what's the expected cost to close the route?
    # Approximate by: if this is the last feasible node, cost = d_current[node] + d_depot[node]
    # For a smoother surrogate, use a soft-min over remaining feasible nodes
    # But for simplicity, use the direct closing cost as a lookahead
    closing_cost = d_current + d_depot
    # Normalize closing cost to be comparable to base_scores
    # Scale by a factor that makes it comparable in magnitude
    closing_penalty = 0.3 * (closing_cost - np.mean(closing_cost))

    # Combine scores: minimize total
    # base_scores: lower is better (more savings)
    # capacity_penalty: lower is better (less demand used)
    # closing_penalty: lower is better (cheaper to close)
    scores = base_scores + capacity_penalty + closing_penalty

    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
