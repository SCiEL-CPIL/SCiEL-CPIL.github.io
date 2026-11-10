import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 325.0  # 5 second safety margin before 330s kill

    # Depot-pressure field model:
    # - Each node has a "pressure" inversely proportional to its demand
    # - High-demand nodes create strong pressure, should be visited early
    # - Edge attractiveness scales with departure node's remaining capacity potential
    #   and arrival node's demand efficiency
    #
    # H[i,j] = (1/dist[i,j]) * (capacity_efficiency[i]) * (demand_pressure[j])
    # where demand_pressure[j] = 1/demand[j] (high demand -> high pressure -> attractive early)
    # and capacity_efficiency[i] reflects how much capacity is "available" at node i
    #
    # For a static heuristic, we approximate capacity_efficiency using the node's
    # own demand relative to capacity: nodes with lower demand leave more room,
    # making them better departure points for subsequent visits.

    # Demand pressure: high-demand nodes are harder to fit, so they should be prioritized
    # Use 1/(1 + demand) as a smooth inverse demand weighting
    demand_pressure = 1.0 / (1.0 + demands)

    # Capacity efficiency: how much remaining capacity a node "represents"
    # Lower demand nodes preserve more capacity, making them good for later in route
    # We use (capacity - demand) / capacity as a proxy for remaining capacity potential
    cap_efficiency = (capacity - demands) / capacity

    # Base distance heuristic
    base_dist = 1.0 / (distance_matrix + 1e-9)

    # Depot-pressure heuristic: combine distance, demand pressure, and capacity efficiency
    # H[i,j] = base_dist[i,j] * cap_efficiency[i] * demand_pressure[j]
    # This makes edges to high-demand nodes more attractive (they need to be done early)
    # and edges from low-demand nodes more attractive (they preserve capacity)
    h_pressure = base_dist * cap_efficiency[:, None] * demand_pressure[None, :]

    # Alternative: symmetric pressure model
    # H[i,j] = base_dist[i,j] * sqrt(cap_efficiency[i] * cap_efficiency[j]) * demand_pressure[j]
    h_sym_pressure = base_dist * np.sqrt(cap_efficiency[:, None] * cap_efficiency[None, :]) * demand_pressure[None, :]

    # Alternative: stronger demand pressure with exponent
    demand_pressure_strong = 1.0 / (1.0 + demands) ** 1.5
    h_pressure_strong = base_dist * cap_efficiency[:, None] * demand_pressure_strong[None, :]

    # Alternative: distance exponent combined with pressure
    base_dist_12 = 1.0 / (distance_matrix ** 1.2 + 1e-9)
    h_pressure_12 = base_dist_12 * cap_efficiency[:, None] * demand_pressure[None, :]

    # Alternative: pure demand clustering - favor edges between similar-demand nodes
    # This encourages routes that group high-demand or low-demand customers together
    demand_sim = 1.0 / (1.0 + np.abs(demands[:, None] - demands[None, :]))
    h_cluster = base_dist * demand_sim * demand_pressure[None, :]

    # Alternative: combined pressure and clustering
    h_combined = base_dist * cap_efficiency[:, None] * demand_pressure[None, :] * (0.5 + 0.5 * demand_sim)

    candidates = [h_pressure, h_sym_pressure, h_pressure_strong, h_pressure_12, h_cluster, h_combined]

    # Fast greedy evaluator with vectorized operations
    def eval_heuristic(h):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        remaining_cap = capacity

        for _ in range(n - 1):
            # Vectorized selection of feasible nodes
            feasible_mask = (~visited[1:]) & (demands[1:] <= remaining_cap)
            if not np.any(feasible_mask):
                # Return to depot
                total_dist += distance_matrix[current, 0]
                current = 0
                remaining_cap = capacity
                feasible_mask = ~visited[1:]

            if not np.any(feasible_mask):
                break

            # Get values for feasible nodes
            feasible_indices = np.where(feasible_mask)[0] + 1
            values = h[current, feasible_indices]

            # Pick best
            best_idx_in_feasible = np.argmax(values)
            best_j = feasible_indices[best_idx_in_feasible]

            total_dist += distance_matrix[current, best_j]
            remaining_cap -= demands[best_j]
            current = best_j
            visited[best_j] = True

        total_dist += distance_matrix[current, 0]
        return total_dist

    best_h = candidates[0]
    best_cost = float('inf')

    for h in candidates:
        if time.time() > deadline:
            break
        cost = eval_heuristic(h)
        if cost < best_cost:
            best_cost = cost
            best_h = h

    # Ensure positive values
    best_h = np.where(best_h > 0, best_h, 1e-9)

    return best_h
