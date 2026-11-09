import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Candidate heuristics
    # 1. Standard inverse distance
    h1 = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Inverse distance weighted by inverse demand (favor low demand)
    # demands is 1D array of length n
    # We want H[i,j] to be low if demand[j] is high.
    # H[i,j] = 1 / (dist[i,j] * (1 + demand[j]))
    demand_factor = 1.0 + demands
    h2 = 1.0 / (distance_matrix * demand_factor[None, :] + 1e-9)
    
    # 3. Mild demand weighting
    h3 = 1.0 / (distance_matrix * (1.0 + 0.5 * demands)[None, :] + 1e-9)
    
    # 4. Additive demand penalty
    h4 = 1.0 / (distance_matrix + demands[None, :] + 1e-9)
    
    # 5. Capacity normalized demand
    # If demand is close to capacity, it's a "full" trip.
    h5 = 1.0 / (distance_matrix * (1.0 + demands / capacity)[None, :] + 1e-9)

    candidates = [h1, h2, h3, h4, h5]
    
    # Fast greedy evaluator
    def eval_heuristic(h):
        visited = [False] * n
        visited[0] = True
        total_dist = 0.0
        current = 0
        remaining_cap = capacity
        
        for _ in range(n - 1):
            best_j = -1
            best_val = -1.0
            
            for j in range(1, n):
                if not visited[j] and demands[j] <= remaining_cap:
                    val = h[current, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
            
            if best_j == -1:
                # Return to depot
                total_dist += distance_matrix[current, 0]
                current = 0
                remaining_cap = capacity
                # Find next node again
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_cap:
                        val = h[current, j]
                        if val > best_val:
                            best_val = val
                            best_j = j
                if best_j == -1:
                    # Fallback: pick any feasible
                    for j in range(1, n):
                        if not visited[j] and demands[j] <= remaining_cap:
                            best_j = j
                            break
                    if best_j == -1:
                        break
            
            total_dist += distance_matrix[current, best_j]
            remaining_cap -= demands[best_j]
            current = best_j
            visited[best_j] = True
        
        total_dist += distance_matrix[current, 0]
        return total_dist

    best_h = h1
    best_cost = float('inf')
    
    for h in candidates:
        cost = eval_heuristic(h)
        if cost < best_cost:
            best_cost = cost
            best_h = h
            
    # Ensure positive values
    best_h = np.where(best_h > 0, best_h, 1e-9)
    
    return best_h
