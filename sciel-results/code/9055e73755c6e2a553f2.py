"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    Design:
    - Uses a blend of prize and effective distance terms.
    - Effective distance D_eff[i, j] = d[i, j] + d[j, 0] accounts for the cost
      of moving from i to j and then returning to the origin.
    - Higher exponents on distance create sharper local attraction, helping the
      best ant find compact high-value routes.
    - Higher exponents on prize emphasize high-value targets.
    - Weights are assigned to favor the sharper (higher exponent) terms.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Handle edge case if matrix is empty or 1x1
    if distance_matrix.size == 0:
        return distance_matrix
    
    n = distance_matrix.shape[0]
    
    # Prize vector broadcast to rows: shape (1, N)
    p = prizes[None, :]
    
    # Distance from i to j: shape (N, N)
    d_ij = distance_matrix
    
    # Distance from j to 0: shape (N, 1)
    d_j0 = distance_matrix[:, 0][:, None]
    
    # Effective distance: cost of i->j->0 loop
    # Shape (N, N)
    D_eff = d_ij + d_j0
    
    # Avoid division by zero
    eps = 1e-9
    D_safe = D_eff + eps
    d_safe = d_ij + eps
    
    # Calculate variants
    # Variant 1: p / D^3
    v1 = p / (D_safe ** 3)
    # Variant 2: p^2 / D^3
    v2 = (p ** 2) / (D_safe ** 3)
    # Variant 3: p / D^4
    v3 = p / (D_safe ** 4)
    # Variant 4: p^2 / D^4
    v4 = (p ** 2) / (D_safe ** 4)
    
    # Blend with heavier weight on higher-exponent terms (v2, v4)
    # Weights sum to 1.0
    w1 = 0.1
    w2 = 0.4
    w3 = 0.1
    w4 = 0.4
    
    heuristic = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    
    # Ensure strictly positive values
    heuristic = heuristic + eps
    
    return heuristic
# EVOLVE-BLOCK-END
