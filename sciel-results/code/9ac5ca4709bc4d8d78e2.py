"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix using Constraint-Aware Value Density.
    
    The heuristic favors moves to nodes with high prize and low "cost to visit and return".
    Cost is defined as d[i, j] + d[j, 0].
    Moves that exceed the max_len budget are heavily penalized (set to minimum).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # d[i, j]
    d_ij = distance_matrix
    # d[j, 0] for all j, broadcast to rows
    d_j0 = distance_matrix[:, 0]
    
    # Cost of moving i -> j and then returning to depot from j
    # This is the marginal cost added to the path length if we go i->j->0
    # The feasibility check in ACO is: current_len + d_ij + d_j0 <= max_len
    # So if d_ij + d_j0 > max_len, the move is infeasible regardless of current_len.
    cost = d_ij + d_j0[None, :]
    
    # Initialize heuristic with minimum value
    heuristic = np.full((n, n), 1e-9)
    
    # Identify feasible edges
    # We allow a tiny epsilon for floating point
    feasible = cost <= (max_len + 1e-9)
    
    # Calculate Value Density: Prize / Cost^2
    # Using Cost^2 strongly penalizes long detours, encouraging "tight" tours
    with np.errstate(divide='ignore', invalid='ignore'):
        values = prizes[None, :] / (cost ** 2)
        
    # Apply feasibility mask
    values[~feasible] = 0.0
    
    # Do not allow moving to self
    values[np.diag_indices(n)] = 0.0
    
    # Normalize so that the max heuristic value is 1.0
    # This balances with pheromone which starts at 1.0
    max_val = values.max()
    if max_val > 0:
        values /= max_val
    else:
        # Fallback if all prizes are 0 or all infeasible
        values[:] = 1e-9
        
    # Ensure all entries are at least 1e-9
    heuristic = np.maximum(values, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
