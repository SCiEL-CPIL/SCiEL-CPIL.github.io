"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using additive penalties and stochasticity.
    
    This approach adds a penalty to the cost of edges in the current tour.
    The penalty is proportional to the number of times the edge has been used
    in previous local optima. A small random noise is added to the penalized
    edges to break symmetry and encourage exploration of new regions.
    """
    n = edge_distance.shape[0]
    perturbed = edge_distance.copy()
    
    # Get the edges in the current local optimal tour
    tour = local_opt_tour
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u < v:
            tour_edges.append((u, v))
        else:
            tour_edges.append((v, u))
            
    # Apply additive penalty to each edge in the tour
    # alpha controls the strength of the penalty.
    # A value of 0.1 means an edge used 10 times gets a penalty of 1.0,
    # which is comparable to the length of an edge in the unit square.
    alpha = 0.1
    
    for (u, v) in tour_edges:
        count = edge_n_used[u, v]
        if count > 0:
            penalty = alpha * count
            # Add the penalty
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty
            # Add small random noise to break symmetry
            noise = np.random.uniform(0, 0.01)
            perturbed[u, v] += noise
            perturbed[v, u] += noise
    
    # Ensure all values are finite and non-negative
    perturbed = np.maximum(perturbed, 0.0)
    perturbed = np.where(np.isfinite(perturbed), perturbed, 1.0)
    
    return perturbed
# EVOLVE-BLOCK-END
