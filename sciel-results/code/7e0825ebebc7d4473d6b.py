"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix for ACO TSP.
    
    Uses inverse-cubed distance as the heuristic. This provides an even
    stronger guidance toward short edges than the inverse-squared version,
    which previously showed significant improvement over the inverse distance
    seed. The assumption is that for uniform random points, short edges are
    highly likely to be in the optimal tour, so maximizing their attractiveness
    helps the ants construct better tours quickly.
    """
    start_time = time.time()
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Safety check for deadline, though this computation is very fast
    if time.time() - start_time > 290:
        # Fallback to simple inverse distance if we are somehow slow (unlikely)
        return 1.0 / (d + 1e-9)

    # Replace diagonal (self-loops) with a large value to make their heuristic ~0
    # We use a large value so 1/d^3 is small.
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Inverse cubed distance: sharper contrast
    # Adding a small epsilon to avoid division by zero for any potential 0s off-diagonal
    # (though distinct random points should have d > 0)
    eta = 1.0 / (d_safe ** 3 + 1e-18)
    
    # Ensure all values are positive and finite
    # If any value is not finite or <= 0, set to 1e-9
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
