"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization
    on a Capacitated Vehicle Routing Problem (CVRP) instance.
    
    Uses a hybrid heuristic combining inverse distance (greedy proximity)
    and a savings-based term (Clarke-Wright principle) to encourage
    routes that minimize total distance by avoiding unnecessary returns to the depot.
    """
    # Ensure inputs are numpy arrays for fast vectorized operations
    D = np.asarray(distance_matrix, dtype=np.float64)
    
    # Safety: Ensure we have a square matrix
    if D.shape[0] != D.shape[1]:
        # Fallback to simple inverse distance if shape is unexpected, though task says same shape
        return 1.0 / (D + 1e-9)

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
    
    # 3. Demand-based term
    # Prefer connecting nodes with higher combined demand (to fill capacity better)
    # This encourages dense routes.
    demand_arr = np.asarray(demands, dtype=np.float64)
    demand_i = demand_arr[:, np.newaxis]
    demand_j = demand_arr[np.newaxis, :]
    combined_demand = demand_i + demand_j
    
    # Normalize combined demand to [0, 1]
    max_combined = np.max(combined_demand)
    if max_combined > 0:
        demand_ratio = combined_demand / max_combined
    else:
        demand_ratio = np.ones_like(combined_demand)
    
    # Combine terms
    # We use a multiplicative boost: Greedy * (1 + Savings) * (1 + alpha * Demand)
    # This ensures that even if savings is low, the greedy term dominates (nearest neighbor).
    # If savings is high, the attractiveness is significantly increased.
    # If demand is high, it further boosts to encourage filling the vehicle.
    alpha = 0.5
    heuristic = term_greedy * (1.0 + savings_ratio) * (1.0 + alpha * demand_ratio)
    
    # Task requirement: "Heuristic entries that are not positive become 1e-9"
    # Also ensures no NaNs or Infs propagate.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
