"""CVRP Heuristic: Route-level look-ahead with nearest-neighbor completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def _complete_route_nn(current_node, depot, unvisited, rest_cap, demands, dist_mat):
    """Simulate completing the current route using nearest-neighbor with capacity check.
    
    Returns the total additional distance to complete this route and return to depot.
    """
    total_dist = 0.0
    cur = current_node
    cap = rest_cap
    remaining = list(unvisited)
    
    while remaining:
        # Find nearest feasible node
        best_node = -1
        best_dist = float('inf')
        for node in remaining:
            if demands[node] <= cap:
                d = dist_mat[cur, node]
                if d < best_dist:
                    best_dist = d
                    best_node = node
        
        if best_node == -1:
            # No feasible node, close route
            total_dist += dist_mat[cur, depot]
            break
        
        total_dist += best_dist
        cur = best_node
        cap -= demands[best_node]
        remaining.remove(best_node)
    
    # Return to depot
    total_dist += dist_mat[cur, depot]
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Route-level look-ahead heuristic: for each feasible candidate, simulate the
    completion of the current route using nearest-neighbor and select the candidate
    that minimizes the total route length.
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
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    n_feasible = len(feasible_nodes)
    
    # For small number of feasible nodes, do full route-level look-ahead
    # For larger sets, use a faster scoring heuristic
    if n_feasible <= 15:
        # Full route-level look-ahead
        best_score = float('inf')
        best_node = int(depot)
        
        # Option: close the route now
        close_score = distance_matrix[current_node, depot]
        
        for j in feasible_nodes:
            # Simulate: visit j, then complete route with NN
            new_cap = rest_capacity - demands[j]
            remaining_after_j = [x for x in feasible_nodes if x != j]
            # Also include infeasible nodes that might become feasible? No, they're still infeasible.
            # Actually, we should simulate with ALL unvisited nodes, not just feasible ones,
            # because some infeasible nodes might be visited later if capacity allows.
            # But wait - infeasible nodes can never be visited since their demand > rest_capacity,
            # and capacity only decreases. So they can never be visited.
            
            # Simulate completion from j with all unvisited nodes (but only feasible ones matter)
            # Actually, we need to be careful: after visiting j, some previously infeasible nodes
            # are still infeasible. So we only consider feasible nodes.
            
            sim_dist = distance_matrix[current_node, j] + _complete_route_nn(
                j, depot, remaining_after_j, new_cap, demands, distance_matrix
            )
            
            if sim_dist < best_score:
                best_score = sim_dist
                best_node = int(j)
        
        # Compare with closing the route
        if close_score < best_score:
            return int(depot)
        else:
            return best_node
    else:
        # Use faster scoring for large candidate sets
        d_current_depot = distance_matrix[current_node, depot]
        d_depot_j = distance_matrix[depot, feasible_nodes]
        d_current_j = distance_matrix[current_node, feasible_nodes]
        demands_feasible = demands[feasible_nodes]
        
        # Savings-based scoring
        savings = d_current_depot + d_depot_j - d_current_j
        
        # Normalize
        max_savings = np.max(savings)
        if max_savings > 0:
            savings_normalized = savings / max_savings
        else:
            savings_normalized = np.zeros_like(savings)
        
        # Demand-weighted proximity
        min_dist = 1e-6
        d_current_j_safe = np.maximum(d_current_j, min_dist)
        alpha = 1.0
        attraction = demands_feasible / (d_current_j_safe ** alpha)
        
        # Normalize attraction
        max_attraction = np.max(attraction)
        if max_attraction > 0:
            attraction_normalized = attraction / max_attraction
        else:
            attraction_normalized = np.zeros_like(attraction)
        
        # Capacity pressure penalty
        total_remaining_demand = np.sum(demands_feasible)
        if total_remaining_demand > 0:
            capacity_pressure = 1.0 - (rest_capacity / max(total_remaining_demand, 1.0))
        else:
            capacity_pressure = 0.0
        capacity_pressure = np.clip(capacity_pressure, 0.0, 1.0)
        
        # Combine scores
        gamma = 0.6
        total_score = gamma * savings_normalized + (1.0 - gamma) * attraction_normalized
        
        # If capacity pressure is very high, prefer to close
        if capacity_pressure > 0.7 and np.max(total_score) < 0.5:
            return int(depot)
        
        max_idx = np.argmax(total_score)
        return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
