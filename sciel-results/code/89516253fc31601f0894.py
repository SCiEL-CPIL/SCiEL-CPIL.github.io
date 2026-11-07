"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Initialize heuristic matrix with zeros
    H = np.zeros((n, n), dtype=float)
    
    # Pre-calculate node feasibility: can we go 0 -> j -> 0 within max_len?
    # If not, the node is useless for the tour.
    # d[0, j] + d[j, 0] <= max_len
    start_end_cost = distance_matrix[0, :] + distance_matrix[:, 0]
    feasible_nodes = start_end_cost <= max_len
    
    # Exponents for prize and distance
    # Higher prize exponent emphasizes high value nodes
    # Higher distance exponent emphasizes proximity
    prize_exp = 2.0
    dist_exp = 1.5
    
    # Calculate heuristic for all pairs
    # We vectorize this for speed
    # Prize component: shape (1, n)
    prize_term = (prizes[None, :] ** prize_exp)
    
    # Distance component: shape (n, n)
    # Avoid division by zero for diagonal
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, np.inf) # Will become 0 after division? No, 1/inf = 0.
    
    dist_term = 1.0 / (dist_safe ** dist_exp)
    
    # Combine: H[i, j] = prize_j * (1 / d[i, j]^dist_exp)
    H = prize_term * dist_term
    
    # Apply feasibility mask
    # If node j is not feasible, H[i, j] should be 0 for all i
    H[:, ~feasible_nodes] = 0.0
    
    # Ensure diagonal is 0 (already handled by inf -> 0, but let's be explicit)
    np.fill_diagonal(H, 0.0)
    
    # Handle non-finite and non-positive values as per task description
    # "Heuristic entries that are not positive become 1e-9"
    # Note: The task says "An exception, non-finite values or a wrong shape make the whole result invalid."
    # So we must ensure NO inf or nan in the output.
    # Our construction avoids inf (by using inf in denominator -> 0) and nan (no 0/0).
    # However, if prize is 0, H is 0. 0 is not positive.
    # So we replace H <= 0 with 1e-9.
    
    H = np.where(H > 0, H, 1e-9)
    
    # Final check for finiteness
    if not np.all(np.isfinite(H)):
        # Fallback to a safe simple heuristic if something went wrong
        H = np.ones((n, n)) * 1e-9
        np.fill_diagonal(H, 1e-9)
        
    return H
# EVOLVE-BLOCK-END
