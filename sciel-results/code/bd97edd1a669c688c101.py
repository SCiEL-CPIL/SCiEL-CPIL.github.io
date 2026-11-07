"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # If no edges have been used yet, return original
    if edge_n_used.max() == 0:
        return edge_distance.copy()
    
    # Determine a large finite value to act as "infinity" for tour edges.
    # It must be significantly larger than any real distance to ensure
    # 2-opt avoids these edges, but finite to prevent numerical issues.
    # Using the max distance in the matrix scaled up ensures it's large enough.
    max_dist = np.max(edge_distance)
    if not np.isfinite(max_dist) or max_dist <= 0:
        max_dist = 1.0
    
    # Set the exclusion value to be much larger than the max distance
    # to force 2-opt to break these edges.
    exclusion_value = max_dist * 10.0 + 1.0
    
    # Extract tour edges as a boolean mask
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = True
        tour_edges[j, i] = True
    
    # Create a copy of the distance matrix
    penalized = edge_distance.copy()
    
    # Apply distance-preserving edge exclusion:
    # Set the distance of all current tour edges to the large exclusion value.
    # This forces the local search to remove these specific "stuck" edges.
    penalized[tour_edges] = exclusion_value
    
    # Ensure finiteness and positivity
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 1e-10)
    
    # Symmetrize to guard against any numerical drift
    penalized = (penalized + penalized.T) / 2.0
    
    return penalized
# EVOLVE-BLOCK-END
