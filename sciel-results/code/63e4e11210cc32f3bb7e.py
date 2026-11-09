"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    Budget-aware expected utility heuristic:
    eta[i,j] = (pi_j / (d[i,j] + d[j,0])^1.5) * exp(-lambda * (d[i,j] + d[j,0]) / max_len)
    
    This explicitly penalizes moves that consume a disproportionate fraction of the
    remaining length budget, preventing ants from getting stuck far from the depot.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Effective cost: d[i,j] + d[j,0]
    # This is the cost of going from i to j and then returning to depot 0
    cost = distance_matrix + distance_matrix[:, 0:1]  # shape (n, n)
    
    # Feasibility mask: edges where cost > max_len are infeasible
    feasible = cost <= max_len
    
    eps = 1e-9
    
    # Budget-aware expected utility
    # Power 1.5 on cost to penalize longer moves more strongly than linear
    # Exponential penalty for consuming large fraction of max_len budget
    lambda_val = 2.0
    
    # Safe cost to avoid division by zero
    cost_safe = cost + eps
    
    # Base: prize[j] / cost[i,j]^1.5
    # High prize and low cost -> high heuristic
    base = prizes[None, :] / (cost_safe ** 1.5)
    
    # Exponential penalty: exp(-lambda * cost / max_len)
    # This smoothly penalizes edges that consume a large fraction of the budget
    # When cost/max_len is large, the penalty is strong
    exp_penalty = np.exp(-lambda_val * cost / max_len)
    
    result = base * exp_penalty
    
    # Apply feasibility: infeasible edges get minimal value
    result = np.where(feasible, result, 1e-9)
    
    # Node 0 has prize 0, so it should naturally be low, but set explicitly
    result[:, 0] = 1e-9
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
