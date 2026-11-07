"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Construct a heuristic matrix that anticipates ACO pheromone dynamics.
    
    The heuristic is a static matrix computed once per instance. We model the
    expected pheromone reinforcement trajectory by simulating a simplified
    ACO-like process with a cooling schedule, then use the resulting
    edge-usage statistics to shape the heuristic.
    
    Key ideas:
    1. Start with inverse-distance as the base heuristic.
    2. Simulate a short ACO-like search with cooling to estimate which edges
       get reinforced.
    3. Combine the base heuristic with the learned edge preferences.
    4. Add demand-awareness and depot-proximity bias.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    
    # Base heuristic: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Compute demand-awareness: penalize edges between high-demand nodes
    # that are unlikely to fit together in a route
    demand_penalty = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                demand_penalty[i, j] = 1.0
                continue
            # If combined demand exceeds capacity, this edge is infeasible
            # in a single route segment; penalize it
            combined = demands[i] + demands[j]
            if combined > capacity:
                demand_penalty[i, j] = 0.1
            else:
                # Mild penalty for high combined demand
                demand_penalty[i, j] = 1.0 / (1.0 + 0.05 * combined)
    
    # Depot proximity bias: edges from/to depot get a slight boost
    depot_bias = np.ones((n, n))
    depot_bias[0, :] = 1.2
    depot_bias[:, 0] = 1.2
    
    # Spatial coherence: prefer edges that go roughly in the same direction
    # from the depot (reduces zig-zagging)
    spatial_coherence = np.ones((n, n))
    depot_coord = coordinates[0]
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            vec_i = coordinates[i] - depot_coord
            vec_j = coordinates[j] - depot_coord
            dot = np.dot(vec_i, vec_j)
            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)
            if norm_i > 1e-9 and norm_j > 1e-9:
                cos_angle = dot / (norm_i * norm_j)
                # Boost edges that go in similar directions
                spatial_coherence[i, j] = 1.0 + 0.15 * max(0.0, cos_angle)
    
    # Simulate a short ACO-like process to learn edge preferences
    # This is a lightweight simulation that runs in < 1 second
    rng = np.random.default_rng(42)
    n_ants_sim = 5
    n_iters_sim = 10
    tau = np.ones((n, n))
    edge_usage = np.zeros((n, n))
    
    for _ in range(n_iters_sim):
        for _ in range(n_ants_sim):
            visited = [0]
            current = 0
            remaining_cap = capacity
            route_cost = 0.0
            
            while True:
                # Find unvisited customers that fit
                candidates = []
                for j in range(1, n):
                    if j not in visited and demands[j] <= remaining_cap:
                        candidates.append(j)
                
                if not candidates:
                    break
                
                # Choose next customer with probability proportional to
                # tau[current, j] * base[current, j] * demand_penalty[current, j]
                weights = np.array([
                    tau[current, j] * base[current, j] * demand_penalty[current, j]
                    for j in candidates
                ])
                total_w = weights.sum()
                if total_w < 1e-12:
                    probs = np.ones(len(candidates)) / len(candidates)
                else:
                    probs = weights / total_w
                
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]
                
                route_cost += distance_matrix[current, next_node]
                edge_usage[current, next_node] += 1.0
                visited.append(next_node)
                remaining_cap -= demands[next_node]
                current = next_node
            
            # Return to depot
            route_cost += distance_matrix[current, 0]
            edge_usage[current, 0] += 1.0
        
        # Update pheromone: decay and add reinforcement
        tau = tau * 0.9 + 1.0 / (1.0 + edge_usage)
    
    # Normalize edge usage to get a preference matrix
    edge_pref = edge_usage / (edge_usage + 1e-9)
    edge_pref = edge_pref / (edge_pref.max() + 1e-9)
    
    # Combine all components
    # Weight the learned preferences with the base heuristic
    learned_weight = 0.3
    heuristic = (1.0 - learned_weight) * base * demand_penalty * depot_bias * spatial_coherence \
                + learned_weight * edge_pref
    
    # Ensure positivity
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
