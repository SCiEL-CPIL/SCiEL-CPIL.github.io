import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 325.0  # 5 second safety margin before 330s kill

    # Generate diverse candidate heuristics
    candidates = []
    
    # 1. Standard inverse distance
    h1 = 1.0 / (distance_matrix + 1e-9)
    candidates.append(h1)
    
    # 2. Distance exponent 1.211 (measured improvement from 1.0)
    h2 = 1.0 / (distance_matrix ** 1.211 + 1e-9)
    candidates.append(h2)
    
    # 3. Distance exponent 1.5 (more discriminative)
    h3 = 1.0 / (distance_matrix ** 1.5 + 1e-9)
    candidates.append(h3)
    
    # 4. Inverse distance weighted by inverse demand
    demand_factor = 1.0 + demands
    h4 = 1.0 / (distance_matrix * demand_factor[None, :] + 1e-9)
    candidates.append(h4)
    
    # 5. Mild demand weighting with exponent
    h5 = 1.0 / (distance_matrix ** 1.2 * (1.0 + 0.5 * demands)[None, :] + 1e-9)
    candidates.append(h5)
    
    # 6. Capacity normalized demand
    h6 = 1.0 / (distance_matrix ** 1.2 * (1.0 + demands / capacity)[None, :] + 1e-9)
    candidates.append(h6)
    
    # 7. Strong demand penalty
    h7 = 1.0 / (distance_matrix * (1.0 + demands)[None, :] + 1e-9)
    candidates.append(h7)
    
    # 8. Additive demand penalty
    h8 = 1.0 / (distance_matrix + demands[None, :] + 1e-9)
    candidates.append(h8)
    
    # Fast greedy evaluator with vectorized operations
    def eval_heuristic(h):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        remaining_cap = capacity
        unvisited = np.arange(1, n)
        
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

    best_h = h1
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
