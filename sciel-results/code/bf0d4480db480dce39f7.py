"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO of CVRP.
    
    Uses Lagrangian relaxation to identify locally optimal edges under relaxed
    capacity constraints, combined with inverse distance, savings, and angular
    proximity signals.
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 300s limit
    
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # 1. Base heuristic: sharpened inverse distance (d^-3)
    base_heuristic = 1.0 / safe_dist ** 3
    
    # 2. Clarke-Wright savings computation
    d0 = distance_matrix[0, 1:]  # shape (n-1,)
    d0_col = d0[:, np.newaxis]  # (n-1, 1)
    d0_row = d0[np.newaxis, :]  # (1, n-1)
    dist_cc = distance_matrix[1:, 1:]  # (n-1, n-1)
    savings_cc = d0_col + d0_row - dist_cc  # (n-1, n-1)
    savings_cc = np.maximum(savings_cc, 0.0)
    
    max_savings = np.max(savings_cc) if savings_cc.size > 0 else 1.0
    if max_savings > 0:
        savings_norm = savings_cc / max_savings
    else:
        savings_norm = savings_cc
    
    alpha_s = 2.0
    savings_boost_cc = 1.0 + alpha_s * savings_norm
    
    savings_boost = np.ones((n, n))
    savings_boost[1:, 1:] = savings_boost_cc
    
    # 3. Demand-aware bias
    max_demand = np.max(demands) if n > 1 else 1.0
    if max_demand > 0:
        demand_norm = demands / max_demand
    else:
        demand_norm = demands
    
    demand_factor = 1.0 + 0.2 * demand_norm
    demand_factor[0] = 1.0
    demand_boost = np.ones((n, n))
    for j in range(1, n):
        demand_boost[:, j] = demand_factor[j]
    
    # 4. Angular proximity bonus
    depot = coordinates[0]
    if n > 1:
        dx = coordinates[1:, 0] - depot[0]
        dy = coordinates[1:, 1] - depot[1]
        angles = np.arctan2(dy, dx)
        
        angles_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
        alpha_a = 0.3
        angular_bonus_cc = 1.0 + alpha_a * np.cos(angles_diff)
        
        angular_bonus = np.ones((n, n))
        angular_bonus[1:, 1:] = angular_bonus_cc
    else:
        angular_bonus = np.ones((n, n))
    
    # 5. Lagrangian relaxation-based edge weight heuristic
    # Solve relaxed CVRP: minimize total distance + lambda * sum(max(0, load - capacity))
    # Use subgradient method to find dual variable lambda that identifies
    # locally optimal edges under relaxed capacity constraints.
    
    # Initialize Lagrangian multiplier
    lambda_lag = 0.0
    lambda_min = 0.0
    lambda_max = 10.0
    
    # Precompute demand-based route costs for each potential edge
    # For each customer j, compute the "effective cost" of visiting j
    # considering demand penalty under Lagrangian relaxation
    
    # Subgradient iteration to find optimal lambda
    best_lambda = 0.0
    best_relaxed_cost = np.inf
    best_edge_weights = None
    
    num_lagrangian_iters = 50
    step_size = 0.1
    
    for lag_iter in range(num_lagrangian_iters):
        # Check deadline
        if time.time() > deadline - 30.0:
            break
        
        # Compute relaxed problem: for each edge (i,j), compute effective cost
        # dist[i,j] + lambda * max(0, cumulative_load_excess)
        
        # Simplified approach: compute node-wise demand penalty
        # For each node j, the marginal cost of adding it to a route
        # depends on how much capacity would be used
        
        # Effective edge cost: dist[i,j] + lambda * demand[j] * 0.1
        # This approximates the Lagrangian penalty for capacity violations
        
        eff_dist = distance_matrix + lambda_lag * demands[np.newaxis, :] * 0.1
        
        # Find shortest paths in the relaxed graph
        # Use Dijkstra-like approach for each node
        relaxed_weights = np.zeros((n, n))
        
        for i in range(n):
            # Compute shortest path from i to all others in relaxed graph
            # Using a simple greedy approach for speed
            unvisited = set(range(n))
            dist_from_i = np.full(n, np.inf)
            dist_from_i[i] = 0.0
            prev = [-1] * n
            
            for _ in range(n):
                if time.time() > deadline - 30.0:
                    break
                
                # Find unvisited node with minimum distance
                min_dist = np.inf
                min_node = -1
                for node in unvisited:
                    if dist_from_i[node] < min_dist:
                        min_dist = dist_from_i[node]
                        min_node = node
                
                if min_node == -1:
                    break
                
                unvisited.remove(min_node)
                
                # Update distances
                for neighbor in unvisited:
                    new_dist = dist_from_i[min_node] + eff_dist[min_node, neighbor]
                    if new_dist < dist_from_i[neighbor]:
                        dist_from_i[neighbor] = new_dist
                        prev[neighbor] = min_node
            
            # Extract edge weights from shortest paths
            for j in range(n):
                if i != j and prev[j] != -1:
                    # Edge prev[j] -> j is part of shortest path
                    relaxed_weights[prev[j], j] += 1.0
        
        # Normalize relaxed weights
        max_rw = np.max(relaxed_weights) if np.max(relaxed_weights) > 0 else 1.0
        relaxed_weights_norm = relaxed_weights / max_rw
        
        # Compute total relaxed cost
        total_relaxed_cost = np.sum(eff_dist[np.arange(n), np.argmin(eff_dist, axis=1)])
        
        if total_relaxed_cost < best_relaxed_cost:
            best_relaxed_cost = total_relaxed_cost
            best_lambda = lambda_lag
            best_edge_weights = relaxed_weights_norm.copy()
        
        # Update lambda using subgradient
        # Gradient: sum of demand excesses (approximate)
        # If average load per route exceeds capacity, increase lambda
        avg_load = np.mean(demands[1:])
        if avg_load > capacity / 50:  # Rough estimate
            grad = avg_load - capacity / 50
        else:
            grad = -0.1
        
        lambda_lag = np.clip(lambda_lag + step_size * grad, lambda_min, lambda_max)
        step_size *= 0.95  # Decay step size
    
    # Use best Lagrangian edge weights if available
    if best_edge_weights is not None:
        lagrangian_boost = 1.0 + 1.5 * best_edge_weights
    else:
        lagrangian_boost = np.ones((n, n))
    
    # Combine all components
    result = base_heuristic * savings_boost * demand_boost * angular_bonus * lagrangian_boost
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    result = np.maximum(result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
