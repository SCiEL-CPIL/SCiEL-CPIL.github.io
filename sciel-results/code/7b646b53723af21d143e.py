"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization
    on a Capacitated Vehicle Routing Problem (CVRP) instance.
    
    Uses a refined hybrid heuristic combining inverse distance (greedy proximity)
    and a savings-based term (Clarke-Wright principle) to encourage
    routes that minimize total distance by avoiding unnecessary returns to the depot.
    """
    # Ensure inputs are numpy arrays for fast vectorized operations
    D = np.asarray(distance_matrix, dtype=np.float64)
    
    # Safety: Ensure we have a square matrix
    if D.shape[0] != D.shape[1]:
        # Fallback to simple inverse distance if shape is unexpected, though task says same shape
        D_safe = D + 1e-9
        return 1.0 / D_safe

    n = D.shape[0]
    depot = 0
    
    # Distances from all nodes to the depot
    d_depot = D[depot, :].copy()
    
    # 1. Inverse Distance Term (Greedy Component)
    # Larger value for closer nodes.
    # Add epsilon to avoid division by zero.
    D_safe = D + 1e-9
    term_greedy = 1.0 / D_safe
    
    # 2. Savings Term (Structure Component)
    # Savings S_ij = d(i,0) + d(j,0) - d(i,j)
    # High savings indicate that connecting i and j is "good" because it 
    # avoids going back to the depot twice.
    
    # Vectorized calculation of d_i0 + d_j0
    # d_i0 is column vector (n, 1), d_j0 is row vector (1, n)
    d_i0 = d_depot[:, np.newaxis]
    d_j0 = d_depot[np.newaxis, :]
    sum_d_to_depot = d_i0 + d_j0
    
    # Avoid division by zero for the (0,0) entry
    sum_d_safe = sum_d_to_depot + 1e-9
    
    # Raw savings
    savings = sum_d_to_depot - D
    
    # Normalized savings ratio: How much of the "potential" savings is realized?
    # Ratio is 1 if D_ij is 0 (perfect connection, i=j)
    # Ratio is 0 if D_ij = d_i0 + d_j0 (worst case, straight line through depot)
    # Ratio is negative if triangle inequality is violated (should not happen in Euclidean)
    savings_ratio = savings / sum_d_safe
    
    # Clip to [0, 1] to ensure we only boost, not penalize, and handle any numerical noise
    savings_ratio = np.clip(savings_ratio, 0.0, 1.0)
    
    # Combine terms with a slightly different weighting strategy
    # We use a multiplicative boost: Greedy * (1 + 0.8 * Savings)
    # This slightly dampens the savings term to let the greedy component
    # have a bit more influence, which can sometimes lead to tighter routes.
    heuristic = term_greedy * (1.0 + 0.8 * savings_ratio)
    
    # Task requirement: "Heuristic entries that are not positive become 1e-9"
    # Also ensures no NaNs or Infs propagate.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
