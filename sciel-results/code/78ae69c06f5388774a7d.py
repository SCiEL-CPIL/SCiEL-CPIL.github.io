"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    # Explicit deadline guard for safety, though calculation is fast
    start_time = time.time()
    deadline = start_time + 1.0
    
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Check time, return safe fallback if exceeded (unlikely for N=50)
    if time.time() > deadline:
        H = np.ones((n, n)) * 1e-9
        np.fill_diagonal(H, 1e-9)
        return H

    # Pre-calculate node feasibility: minimum cost to visit j and return is d[0, j] + d[j, 0]
    # If this exceeds max_len, the node can never be part of a valid tour starting/ending at 0.
    min_cost_j = d[0, :] + d[:, 0]
    feasible_nodes = min_cost_j <= max_len
    
    # Calculate Detour Cost
    # Detour[i, j] = d[i, j] + d[j, 0] - d[i, 0]
    # This represents the additional path length incurred by visiting j from i, 
    # compared to the direct return from i to 0.
    
    d_j0 = d[:, 0] # Shape (n,)
    d_i0 = d[:, 0] # Shape (n,)
    
    # Broadcasting:
    # d is (n, n)
    # d_j0[None, :] is (1, n)
    # d_i0[:, None] is (n, 1)
    detour = d + d_j0[None, :] - d_i0[:, None]
    
    # Ensure non-negative (numerical stability)
    detour = np.maximum(detour, 0.0)
    
    # Structural Change: Fractional Heuristic
    # H[i, j] = (Prize[j] / (Prize[j] + Detour[i, j]))^alpha
    # This balances prize value against the incremental cost (detour).
    # It is scale-invariant to some extent.
    
    # Handle Prize[j] == 0
    # If Prize[j] is 0, H is 0.
    # If Detour is 0 and Prize is 0, 0/0.
    # We add a small epsilon to denominator or handle separately.
    
    alpha = 2.0
    
    # Numerator: Prize[j]^alpha
    # Shape (1, n)
    num = (prizes[None, :] ** alpha)
    
    # Denominator: (Prize[j] + Detour[i, j])^alpha
    # Shape (n, n)
    denom = (prizes[None, :] + detour) ** alpha
    
    # Avoid division by zero
    # If num is 0 and denom is 0, we want 0 (or 1e-9 later).
    # If num is 0, H is 0.
    # If denom is 0 (Prize=0, Detour=0), H is 0/0.
    # Let's use np.divide with where clause.
    
    H = np.zeros((n, n), dtype=float)
    mask = denom > 0
    np.divide(num, denom, out=H, where=mask)
    
    # Apply feasibility mask
    # If node j is not feasible, H[:, j] = 0
    H[:, ~feasible_nodes] = 0.0
    
    # Diagonal should be 0 (don't visit current node)
    np.fill_diagonal(H, 0.0)
    
    # Handle non-positive and non-finite values
    # "Heuristic entries that are not positive become 1e-9"
    H = np.where(H > 0, H, 1e-9)
    
    # Final finiteness check
    if not np.all(np.isfinite(H)):
        H = np.ones((n, n)) * 1e-9
        np.fill_diagonal(H, 1e-9)
        
    return H
# EVOLVE-BLOCK-END
