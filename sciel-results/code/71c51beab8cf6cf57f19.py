"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Ensure valid inputs
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    # Compute the "return cost" for each node: d[j, 0]
    # This is the cost to go back to the start after visiting j
    return_cost = distance_matrix[:, 0].copy()
    
    # For each pair (i, j), the "effective distance" is d[i,j] + d[j,0]
    # This represents the total additional cost if we go from i to j and then must return to start
    # But actually, the constraint is: length_so_far + d[i,j] + d[j,0] <= max_len
    # So the "marginal cost" of adding j after i is d[i,j] + d[j,0]
    
    # However, we want a heuristic that favors:
    # 1. High prize nodes
    # 2. Nodes that are "cheap" to add (low d[i,j] + d[j,0])
    # 3. Nodes that leave room for more high-value nodes later
    
    # Let's compute a reachability-weighted prize
    # The key insight: a node j is more attractive if:
    # - It has a high prize
    # - It's close to the current node i (low d[i,j])
    # - It doesn't consume too much of the remaining budget (low d[j,0])
    
    # A good heuristic: prize[j] * exp(-beta * (d[i,j] + d[j,0]))
    # This exponentially penalizes nodes that are far or have high return cost
    
    # Let's try a combination:
    # heuristic[i,j] = prize[j]^alpha / (d[i,j] + d[j,0] + epsilon)^beta
    
    # But we need to be careful: d[i,j] + d[j,0] could be very small for nearby nodes
    # which would make the heuristic very large, which is good
    
    # Let's use:
    # - alpha = 2 to emphasize high prizes
    # - beta = 1.5 to penalize distance/return cost
    # - epsilon = 1e-6 to avoid division by zero
    
    epsilon = 1e-6
    alpha = 2.0
    beta = 1.5
    
    # prize_term[j] = prize[j]^alpha
    prize_term = np.power(prizes + epsilon, alpha)
    
    # cost_term[i,j] = (d[i,j] + d[j,0] + epsilon)^beta
    cost_term = np.power(distance_matrix + return_cost[None, :] + epsilon, beta)
    
    # heuristic[i,j] = prize_term[j] / cost_term[i,j]
    heuristic = prize_term[None, :] / cost_term
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
