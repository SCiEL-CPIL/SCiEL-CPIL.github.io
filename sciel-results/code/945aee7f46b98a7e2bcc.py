# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO of capacitated vehicle routing.
    Uses a powered inverse-distance heuristic to strongly prefer short edges.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # Create a copy to avoid modifying the input
    d = dm.copy()
    
    # Set diagonal to a large value so self-loops are never chosen
    np.fill_diagonal(d, np.inf)
    
    # Use powered inverse distance: eta = 1 / d^gamma
    # gamma=2 strongly prefers short edges, which works well for
    # 50-customer instances in the unit square
    gamma = 2.0
    
    # Compute heuristic with numerical safety
    # Replace any zero distances (other than diagonal) with a small value
    d_safe = np.where(d > 1e-9, d, 1e-9)
    
    eta = 1.0 / (d_safe ** gamma)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Set diagonal back to a very small value (should never be selected)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
