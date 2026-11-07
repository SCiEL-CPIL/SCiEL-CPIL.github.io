import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses a Gaussian decay heuristic based on the cost-to-prize ratio.
    H_ij = exp(- (d[i,j] + d[j,0]) / (sigma * prize[j]))
    This structurally differs from power-law by using exponential decay,
    creating a sharper focus on high-value, low-cost nodes.
    """
    n = len(prizes)
    d = distance_matrix
    d_to_0 = d[:, 0]
    
    # Effective cost: distance from i to j plus distance from j to 0
    C = d + d_to_0[None, :]
    
    # Ensure prizes are positive to avoid division by zero in the exponent
    # If prize is 0, the heuristic should be effectively 0 (or 1e-9)
    prizes_safe = np.maximum(prizes, 1e-6)
    
    # Sigma controls the "width" of the Gaussian decay.
    # Smaller sigma means steeper decay, favoring only very high prize/cost ratios.
    sigma = 0.3
    
    # Calculate the cost-to-prize ratio
    # Shape (n, n)
    ratio = C / (sigma * prizes_safe[None, :])
    
    # Apply Gaussian decay
    heuristic = np.exp(-ratio)
    
    # Handle edge cases: non-finite or non-positive values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
