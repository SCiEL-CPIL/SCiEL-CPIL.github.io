"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using additive penalties.
    
    This approach adds a penalty to the cost of edges in the current local
    optimal tour. The penalty is proportional to the number of times the edge
    has appeared in local optima (stored in edge_n_used). This encourages
    the local search (2-opt) to break these "frozen" edges and explore
    different parts of the solution space.
    """
    n = edge_distance.shape[0]
    perturbed = edge_distance.copy()
    
    # Calculate a base penalty scale based on the average edge cost
    # This ensures the penalty is relative to the problem scale.
    avg_cost = np.mean(edge_distance)
    if avg_cost == 0:
        avg_cost = 1.0
        
    # Base penalty factor. 
    # We scale by (1 + count) so that frequently used edges get larger penalties.
    # 0.5 means if an edge is used once, penalty is 0.5 * avg_cost * 2 = 1.0 * avg_cost.
    # This effectively doubles the cost of edges in the first local optimum found.
    base_factor = 0.5
    
    # Get the edges in the current local optimal tour
    tour = local_opt_tour
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        
        # Get usage count for this edge
        count = edge_n_used[u, v]
        
        # Calculate penalty
        # We add this to both [u,v] and [v,u]
        penalty = base_factor * avg_cost * (1 + count)
        
        # Apply additive penalty
        perturbed[u, v] += penalty
        perturbed[v, u] += penalty
        
    # Ensure all values are finite and non-negative
    # Additive penalties on positive distances keep them positive.
    # Check for any numerical issues just in case.
    perturbed = np.maximum(perturbed, 0.0)
    perturbed = np.where(np.isfinite(perturbed), perturbed, 1.0)
    
    return perturbed
# EVOLVE-BLOCK-END
