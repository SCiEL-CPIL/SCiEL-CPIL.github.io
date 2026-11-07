"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.
    
    Strategy: marginal gain based on prize density per unit of remaining path budget.
    For each edge (i, j), estimate the slack after visiting j and returning home,
    then weight prize[j] by this slack to prioritize high-value nodes that fit well
    in the remaining budget.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure finite values
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    prizes = np.where(np.isfinite(prizes), prizes, 0.0)
    
    # Build heuristic: for each (i, j), compute marginal gain
    # slack = max_len - d[i,j] - d[j,0] (remaining budget after visiting j and returning)
    # attractiveness ~ prize[j] * max(slack, 0) / (d[i,j] + eps)
    # This rewards nodes with high prize that leave more budget remaining,
    # normalized by travel cost to i->j.
    
    eps = 1e-9
    
    # d[i,j] for all i,j
    dij = distance_matrix  # shape (n, n)
    
    # d[j,0] for all j (return cost)
    d_j_0 = distance_matrix[:, 0]  # shape (n,)
    
    # For each pair (i, j): slack = max_len - dij[i,j] - d_j_0[j]
    # We want: prize[j] * max(slack, 0) / (dij[i,j] + eps)
    
    # Build matrix: shape (n, n)
    # slack[i, j] = max_len - dij[i, j] - d_j_0[j]
    slack = max_len - dij - d_j_0[None, :]  # shape (n, n), broadcast d_j_0 over rows
    slack = np.maximum(slack, 0.0)
    
    # prize[j] broadcast over rows
    prize_j = prizes[None, :]  # shape (1, n)
    
    # denominator: dij[i,j] + eps
    denom = dij + eps
    
    # heuristic[i,j] = prize[j] * slack[i,j] / denom[i,j]
    heuristic = prize_j * slack / denom
    
    # Handle diagonal: no self-loops, set to very small
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Additional sharpening: raise to a power to amplify differences
    # This helps ants converge faster on high-value edges
    heuristic = heuristic ** 1.5
    
    # Final safety: ensure positive finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
