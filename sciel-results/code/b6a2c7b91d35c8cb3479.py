"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Structural Change: Use an exponential cost penalty on the combined step-and-return distance.
    # This models the "commitment" of visiting a node: you pay the step cost d[i,j] 
    # and you commit to paying the return cost d[j,0] eventually.
    # The heuristic rewards high prizes but decays exponentially with this total commitment cost.
    
    d_j0 = distance_matrix[:, 0]  # (n,) return cost for each node j
    
    # Combined cost matrix: cost[i, j] = d[i, j] + d[j, 0]
    # This represents the budget impact of moving i -> j and ensuring we can return from j.
    cost_matrix = distance_matrix + d_j0[None, :]
    
    # Exponential decay parameters
    # alpha controls the emphasis on prize value
    # beta controls the sharpness of the cost penalty
    # With max_len=3 and unit square, max cost is ~2.8.
    # beta=1.5 gives exp(-4.2) ~ 0.015 for max cost, which is a strong penalty.
    alpha = 1.5
    beta = 1.5
    
    # Prize term
    prize_term = np.power(prizes, alpha)
    prize_term = np.maximum(prize_term, 1e-9) # Avoid 0^0 issues if prize is 0 (node 0)
    
    # Cost term: exponential decay
    cost_term = np.exp(-beta * cost_matrix)
    
    heuristic = prize_term[None, :] * cost_term
    
    # Row-normalize to ensure consistent scale across rows
    # This prevents rows with generally higher costs from having systematically lower probabilities
    row_sums = np.sum(heuristic, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-9)
    heuristic = heuristic / row_sums
    
    # Set diagonal to very small (no self-moves)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
