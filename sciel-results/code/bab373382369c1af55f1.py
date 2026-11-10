"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for TSP using a negative-curvature /
    high-power distance penalty to suppress long-range crossing edges.
    
    The heuristic is inversely proportional to distance raised to a high power,
    which aggressively penalizes long edges. This leverages the fact that optimal
    TSP tours in 2D are planar and prefer short, local edges.
    
    We use d^4 (or similar high power) to strongly suppress non-local connections.
    """
    # Ensure we're working with a numpy array
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Guard against zero distances (shouldn't happen with random points, but just in case)
    d_safe = np.maximum(d, 1e-9)
    
    # Use a high power of distance to aggressively suppress long-range edges.
    # d^2 would be moderate; d^4 or d^6 strongly favors very short edges.
    # The directive says "sum of squared edge lengths (or a higher power of distance)"
    # Let's use d^4 to strongly penalize long edges while keeping the heuristic well-behaved.
    # 
    # Actually, let's think about this more carefully. The heuristic is used as:
    # probability ∝ pheromone[i,j] * heuristic[i,j]
    # 
    # If heuristic = 1/d^k, then for large k, the probability becomes extremely biased
    # toward the nearest neighbors. This can cause the ants to get stuck in local patterns
    # that don't form good tours. 
    # 
    # A good balance is around k=2 to k=4. Let's use k=3 as a compromise:
    # strong enough to suppress long crossing edges, but not so strong that it
    # completely locks into nearest-neighbor chains.
    #
    # Actually, re-reading the directive: "inversely proportional to the sum of squared
    # edge lengths (or a higher power of distance)". This suggests using d^2 or higher.
    # Let's use d^3 for a strong but not extreme penalty.
    
    heuristic = 1.0 / (d_safe ** 3)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
