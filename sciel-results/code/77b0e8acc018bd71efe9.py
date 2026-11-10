"""CVRP Heuristic: Savings-based with capacity-aware angular sweep and lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Combined CVRP heuristic: savings-based selection with angular penalty
    and capacity awareness. Uses a 1-step lookahead to evaluate the
    immediate cost of visiting each candidate node.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Extract relevant distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    demands_feasible = demands[feasible_nodes]
    
    # Classic savings: S(C, J) = d(C, Depot) + d(Depot, J) - d(C, J)
    # Higher savings means it's better to visit J now rather than in a separate route
    savings = d_current_depot + d_depot_j - d_current_j
    
    # 1-step lookahead: for each candidate J, estimate the cost of the next step
    # after visiting J. We use the nearest feasible neighbor as a proxy.
    # cost_after_J = d(J, best_next) + d(best_next, depot) if capacity allows
    # Otherwise, cost_after_J = d(J, depot)
    
    # For speed, we approximate the lookahead by considering the best savings
    # from J to any other feasible node (excluding J itself)
    n = len(feasible_nodes)
    
    if n > 1:
        # Build distance matrix among feasible nodes
        d_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        
        # For each J, find the best next node K (K != J) that is feasible
        # and maximizes savings from J
        # savings_from_J_to_K = d(J, depot) + d(depot, K) - d(J, K)
        d_j_depot = d_depot_j  # distance from each feasible node to depot
        
        # savings_matrix[i, j] = savings from going i -> j instead of separate routes
        # = d(i, depot) + d(depot, j) - d(i, j)
        savings_matrix = d_j_depot[:, np.newaxis] + d_j_depot[np.newaxis, :] - d_ff
        
        # Set diagonal to -inf so we don't pick the same node
        np.fill_diagonal(savings_matrix, -np.inf)
        
        # For each J, the best savings to a next node
        best_next_savings = np.max(savings_matrix, axis=1)
        
        # Lookahead cost: if we visit J, the additional cost is roughly
        # d(current, J) + min(d(J, depot), d(J, best_next) + d(best_next, depot))
        # But we approximate: the "effective cost" of visiting J is
        # d_current_j - best_next_savings (higher savings = lower effective cost)
        
        # Capacity check for lookahead: if rest_capacity - demand[J] < min demand of others,
        # we can't visit another node, so we must return to depot
        remaining_cap_after = rest_capacity - demands_feasible
        min_other_demand = np.min(demands_feasible)
        
        # For nodes where we can't visit another node after, the lookahead is just d(J, depot)
        # For others, the lookahead includes the savings
        can_continue = remaining_cap_after >= min_other_demand
        
        # Effective score: savings + lookahead adjustment
        # Higher is better
        lookahead_bonus = np.where(can_continue, best_next_savings * 0.3, 0.0)
        
        # Angular penalty: penalize nodes that are in a similar direction to current
        # relative to depot (to encourage spreading out)
        # Compute angle from depot to current and depot to each feasible node
        # We use a simple proxy: the angle between vectors (current - depot) and (j - depot)
        # Since we don't have coordinates, we use the distance matrix to approximate
        # via the law of cosines: cos(theta) = (d(depot,current)^2 + d(depot,j)^2 - d(current,j)^2) / (2*d(depot,current)*d(depot,j))
        
        d_depot_current = d_current_depot
        d_depot_j_safe = np.maximum(d_depot_j, 1e-6)
        d_current_j_safe = np.maximum(d_current_j, 1e-6)
        
        # cos(theta) = (a^2 + b^2 - c^2) / (2*a*b) where a=d(depot,current), b=d(depot,j), c=d(current,j)
        cos_theta = (d_depot_current**2 + d_depot_j**2 - d_current_j**2) / (2 * d_depot_current * d_depot_j_safe)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Angle in radians
        theta = np.arccos(cos_theta)
        
        # Angular penalty: small when theta is large (node is in a different direction)
        # We want to encourage visiting nodes in different directions to cover the area
        angular_penalty = theta  # Higher theta = less penalty (we subtract this)
        
        # Normalize angular penalty to [0, 1] range approximately
        angular_penalty_normalized = angular_penalty / np.pi
        
        # Combine scores
        # Primary: savings (classic)
        # Secondary: lookahead bonus
        # Tertiary: angular diversity (encourage spreading)
        # Capacity pressure: when rest_capacity is low, prefer to close the route
        
        total_remaining_demand = np.sum(demands_feasible)
        if total_remaining_demand > 0:
            capacity_pressure = 1.0 - (rest_capacity / max(total_remaining_demand, 1.0))
        else:
            capacity_pressure = 0.0
        capacity_pressure = np.clip(capacity_pressure, 0.0, 1.0)
        
        # Normalize savings
        max_savings = np.max(savings)
        min_savings = np.min(savings)
        if max_savings > min_savings:
            savings_normalized = (savings - min_savings) / (max_savings - min_savings)
        else:
            savings_normalized = np.zeros_like(savings)
        
        # Total score
        # Weight savings heavily, add lookahead and angular components
        total_score = (
            1.0 * savings_normalized +
            0.4 * (lookahead_bonus / max_savings if max_savings > 0 else 0) +
            0.3 * angular_penalty_normalized -
            0.5 * capacity_pressure
        )
        
        # If capacity pressure is very high, prefer to return to depot
        if capacity_pressure > 0.7 and np.max(total_score) < 0.3:
            return int(depot)
        
    else:
        # Only one feasible node, visit it
        return int(feasible_nodes[0])
    
    max_idx = np.argmax(total_score)
    
    # Check if it's better to close the route
    # Compare the score of the best node vs. the "cost" of closing
    # Closing cost: d(current, depot)
    # Visiting cost: d(current, J) + estimated future cost
    # If d(current, depot) is much smaller than d(current, J) + future, close
    
    best_node = feasible_nodes[max_idx]
    d_to_best = distance_matrix[current_node, best_node]
    
    # Heuristic: if the best node is far and capacity is tight, close
    if d_to_best > 2.0 * d_current_depot and capacity_pressure > 0.5:
        return int(depot)
    
    return int(best_node)
# EVOLVE-BLOCK-END
