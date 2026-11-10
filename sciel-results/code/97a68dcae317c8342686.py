"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix using Constraint-Aware Value Density
    with smooth surrogate for budget constraints.
    
    Uses a soft-min formulation to smoothly handle the max_len constraint,
    combined with prize density and distance decay.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Precompute return costs: d[j, 0] for all j
    return_costs = distance_matrix[:, 0]
    
    # Initialize heuristic matrix with minimum value
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Parameters
    lambda_budget = 3.0  # Strength of budget penalty (increased for sharper constraint)
    mu_distance = 0.5    # Reduced distance decay to allow more exploration
    epsilon = 1e-6       # Smaller epsilon for better conditioning
    soft_min_temp = 0.1  # Temperature for soft-min approximation
    
    # Vectorized computation for all pairs
    # Cost matrix: cost[i,j] = d[i,j] + d[j,0]
    cost_matrix = distance_matrix + return_costs[None, :]
    
    # Budget feasibility: how much budget is left
    # budget_left[i,j] = max_len - cost[i,j]
    budget_left = max_len - cost_matrix
    
    # Soft-min approximation of min(budget_left, 0) using log-sum-exp trick
    # We want to smoothly penalize infeasible moves
    # For feasible moves (budget_left >= 0), penalty ~ 0
    # For infeasible moves (budget_left < 0), penalty ~ budget_left
    
    # Use smooth max(0, -budget_left) = smooth max(0, cost_matrix - max_len)
    # soft_max(a, b) = log(exp(a) + exp(b)) with scaling
    # For soft_max(0, x) where x = cost_matrix - max_len:
    # soft_max(0, x) = -soft_min(0, -x) = -log(exp(0) + exp(-x)) * temp
    #                 = -temp * log(1 + exp(-x/temp))
    
    x = (cost_matrix - max_len) / soft_min_temp
    # Stable log(1 + exp(-x)) = logaddexp(0, -x)
    soft_penalty = -soft_min_temp * np.logaddexp(0, -x)
    
    # Budget penalty: exp(-lambda_budget * soft_penalty)
    # When feasible (x <= 0), soft_penalty ~ 0, so penalty ~ 1
    # When infeasible (x >> 0), soft_penalty ~ x * temp = cost - max_len, so penalty ~ exp(-lambda*(cost-max_len))
    budget_penalty = np.exp(-lambda_budget * soft_penalty)
    
    # Prize density: prizes[j] / (cost + epsilon)
    # For each j, prizes[j] is the same across all i
    prize_density = prizes[None, :] / (cost_matrix + epsilon)
    
    # Distance decay: exp(-mu * d[i,j])
    distance_decay = np.exp(-mu_distance * distance_matrix)
    
    # Combine all components
    heuristic = prize_density * budget_penalty * distance_decay
    
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Normalize to [1e-9, 1]
    max_val = np.max(heuristic)
    if max_val > 0 and np.isfinite(max_val):
        heuristic = heuristic / max_val
    
    # Ensure all entries are at least 1e-9 and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
