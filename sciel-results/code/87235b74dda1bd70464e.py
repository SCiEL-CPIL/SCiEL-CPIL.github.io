"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for ACO.
    Uses a combination of inverse distance and Clarke-Wright savings boost.
    """
    n = distance_matrix.shape[0]
    # Ensure we don't divide by zero or have issues with diagonal
    # Diagonal is not used in standard ACO transition (i != j), but we set it to 0 or small.
    
    # 1. Base heuristic: Inverse distance
    # Add epsilon to avoid division by zero for i==j, though i!=j is enforced by ACO logic usually.
    # But the matrix is returned fully.
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Savings Heuristic
    # Savings S_ij = d_0i + d_0j - d_ij
    # We want to boost edges that have high savings.
    # d_0 is the first row (or column) of the distance matrix.
    d0 = distance_matrix[0, :]
    
    # Calculate savings matrix
    # S_ij = d0_i + d0_j - d_ij
    savings = d0[:, np.newaxis] + d0[np.newaxis, :] - distance_matrix
    
    # Only positive savings are "good" (they represent a merge that is shorter than separate trips)
    # Negative savings mean the edge is a detour.
    positive_savings = np.maximum(0, savings)
    
    # Normalize savings to be comparable to inv_dist?
    # Max savings is roughly 2 * max(d0). Max inv_dist is 1/min(d).
    # Let's scale savings so that the max positive savings is 1.
    max_sav = np.max(positive_savings)
    if max_sav > 0:
        norm_savings = positive_savings / max_sav
    else:
        norm_savings = positive_savings
    
    # Combine: 
    # eta_ij = inv_dist_ij * (1 + alpha * norm_savings_ij)
    # Alpha controls the weight of the savings boost.
    # If alpha is high, ants prefer "savings-rich" edges even if slightly further.
    alpha = 2.0
    
    heuristic = inv_dist * (1.0 + alpha * norm_savings)
    
    # Handle the depot row/col specifically?
    # For i=0, savings is d0_0 + d0_j - d_0j = 0 + d0_j - d0_j = 0.
    # So for edges from depot, the boost is 0. They rely on inv_dist.
    # This is correct: leaving the depot, we just want to go to a "close" start node, 
    # but the savings logic will guide the *subsequent* moves to form chains.
    
    # Ensure no non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # This usually happens in the ACO loop, but we can ensure our output is valid.
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
