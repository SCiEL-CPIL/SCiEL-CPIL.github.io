"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    This heuristic uses a Savings-Based approach.
    The attractiveness of an edge (i, j) is determined by how much distance
    is saved by traveling directly from i to j instead of returning to the depot
    and then going to j. This encourages the formation of longer, tighter routes.
    
    We combine the savings value with an inverse distance component to ensure
    that close nodes are still preferred, but the savings term provides the
    structural guidance for route construction.
    """
    n = distance_matrix.shape[0]
    
    # Calculate the savings matrix
    # Savings(i, j) = dist(i, 0) + dist(0, j) - dist(i, j)
    # This represents the distance saved by merging the two segments i->0 and 0->j into i->j
    
    # dist(i, 0) is column 0 of distance_matrix
    # dist(0, j) is row 0 of distance_matrix
    # dist(i, j) is distance_matrix
    
    d_i_0 = distance_matrix[:, 0] # Shape (n,)
    d_0_j = distance_matrix[0, :] # Shape (n,)
    
    # Savings matrix shape (n, n)
    # s[i, j] = d_i_0[i] + d_0_j[j] - distance_matrix[i, j]
    savings = d_i_0[:, np.newaxis] + d_0_j[np.newaxis, :] - distance_matrix
    
    # Ensure savings are non-negative (triangle inequality should hold, but numerical errors or invalid inputs might not)
    savings = np.maximum(savings, 0)
    
    # We want the heuristic to be larger for better edges.
    # High savings are good.
    # However, we also want to penalize long distances directly.
    # A common combination is: eta[i, j] = savings[i, j] / distance_matrix[i, j]
    # Or: eta[i, j] = 1 / distance_matrix[i, j] * (1 + savings[i, j] / max_savings)
    
    # Let's use a normalized savings factor.
    # Normalize savings to [0, 1]
    max_savings = np.max(savings)
    if max_savings > 0:
        norm_savings = savings / max_savings
    else:
        norm_savings = np.zeros_like(savings)
    
    # Base inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Combine: 
    # The heuristic should reward edges that have high savings.
    # We use: eta[i, j] = inv_dist[i, j] * (1 + norm_savings[i, j])
    # This keeps inverse distance as the base but boosts it by the relative savings.
    
    heuristic = inv_dist * (1.0 + norm_savings)
    
    # For the depot row (i=0), savings(0, j) = d(0,0) + d(0,j) - d(0,j) = 0.
    # So heuristic[0, j] = inv_dist[0, j] * 1.0. This is standard.
    
    # For the depot column (j=0), we don't really use these for customer selection 
    # (ants return to depot, but the choice is usually handled separately or 
    # by checking if remaining load allows next customer). 
    # However, the matrix must be full.
    # savings(i, 0) = d(i,0) + d(0,0) - d(i,0) = 0.
    # So heuristic[i, 0] = inv_dist[i, 0].
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
