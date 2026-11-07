"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    
    Uses a greedy path simulation approach: for each node j, simulate a greedy
    path starting from j (or from 0 to j), respecting the budget constraint,
    and compute the total prize collected. This provides a more accurate
    estimate of node value than summing all feasible neighbors.
    
    eta[i, j] = prize[j] / (d[i,j] + d[j,0]) * (1 + gamma * greedy_value[j])
    
    where greedy_value[j] is the total prize from a greedy path starting at j
    with remaining budget = max_len - d[j,0].
    """
    n = len(prizes)
    eps = 1e-9
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s limit
    
    # Basic prize/cost ratio
    prize_col = prizes[None, :] + eps  # (1, n)
    dist_to_origin = distance_matrix[:, 0][None, :] + eps  # (1, n)
    cost_matrix = distance_matrix + dist_to_origin  # (n, n)
    
    # Base heuristic: prize / cost
    base_eta = prize_col / cost_matrix  # (n, n)
    
    # Greedy path simulation for each node j
    # For node j, simulate a greedy path:
    # 1. Start at j with remaining budget = max_len - d[j,0]
    # 2. Repeatedly pick the unvisited node k that maximizes prize[k] / d[j,k]
    #    subject to d[current, k] <= remaining_budget
    # 3. Add prize[k] to total, update current = k, remaining_budget -= d[current, k]
    # 4. Stop when no node qualifies or budget exhausted
    
    # Precompute: for each pair (i, j), the cost to go from i to j
    # This is just distance_matrix[i, j]
    
    greedy_values = np.zeros(n)
    
    for start_node in range(n):
        if time.time() > deadline - 10.0:
            break
            
        # Remaining budget after visiting start_node and returning to 0
        remaining = max_len - distance_matrix[start_node, 0]
        if remaining <= 0:
            greedy_values[start_node] = prizes[start_node]
            continue
        
        total_prize = prizes[start_node]
        current = start_node
        visited = set([start_node])
        
        # Greedily add nodes
        while True:
            best_score = -1
            best_node = -1
            
            for k in range(n):
                if k in visited:
                    continue
                d = distance_matrix[current, k]
                if d > remaining:
                    continue
                # Score: prize per unit distance
                score = prizes[k] / (d + eps)
                if score > best_score:
                    best_score = score
                    best_node = k
            
            if best_node == -1:
                break
            
            total_prize += prizes[best_node]
            remaining -= distance_matrix[current, best_node]
            current = best_node
            visited.add(best_node)
            
            if remaining <= 0:
                break
        
        greedy_values[start_node] = total_prize
    
    # Normalize greedy_values to [0, 1]
    gv_max = greedy_values.max()
    if gv_max > eps:
        greedy_norm = greedy_values / gv_max
    else:
        greedy_norm = np.zeros(n)
    
    # Blend: use gamma to control how much greedy path info matters
    gamma = 1.0
    
    # The multiplier for each column j
    col_multiplier = 1.0 + gamma * greedy_norm  # (n,)
    
    # Apply multiplier: eta[i,j] *= col_multiplier[j]
    eta = base_eta * col_multiplier[None, :]  # (n, n)
    
    # Soft feasibility: favor edges with more remaining budget
    remaining = max_len - cost_matrix  # (n, n)
    remaining_clipped = np.clip(remaining, 0, max_len)
    
    if max_len > eps:
        remaining_norm = remaining_clipped / max_len
    else:
        remaining_norm = np.zeros_like(remaining_clipped)
    
    # Apply a gentle boost for edges with more remaining budget
    beta = 0.3
    feasibility_boost = 1.0 + beta * remaining_norm  # (n, n)
    
    eta = eta * feasibility_boost
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Verify shape
    if eta.shape != (n, n):
        eta = np.ones((n, n)) * 1e-9
    
    return eta
# EVOLVE-BLOCK-END
