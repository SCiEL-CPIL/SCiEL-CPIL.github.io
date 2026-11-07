# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Reactive Tabu Search style perturbation for TSP.
    
    This approach uses a 'hard' tabu mechanism: edges that are currently in the tour 
    and have been used more frequently than the median edge are heavily penalized 
    to force their removal. This differs from the standard GLS which applies 
    gradual penalties to all edges based on usage.
    """
    n = edge_distance.shape[0]
    
    # Calculate usage statistics
    # Flatten the upper triangle to get unique edge usages
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    usages = edge_n_used[mask]
    
    if len(usages) > 0:
        median_usage = np.median(usages)
        max_usage = np.max(usages)
    else:
        median_usage = 0
        max_usage = 0

    # Initialize perturbed matrix as a copy of true distances
    perturbed = edge_distance.copy()
    
    # Identify edges in the current tour
    tour_edges = set()
    tour = local_opt_tour
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u < v:
            tour_edges.add((u, v))
        else:
            tour_edges.add((v, u))

    # Apply Reactive Tabu Penalties
    # We penalize edges that are in the current tour and have high usage.
    # The penalty is a large multiplier to effectively make these edges "tabu".
    
    # Determine a threshold for "high" usage. 
    # Using the median ensures we target the most frequently used edges in the tour.
    # If max usage is 0, no penalties.
    if max_usage == 0:
        return perturbed

    # Penalty factor:
    # For edges in the tour with usage > median_usage, we apply a strong penalty.
    # We use a multiplier of 5.0 to strongly discourage 2-opt from keeping them.
    # For other edges, we leave them alone or apply a very small penalty to maintain diversity.
    
    penalty_strength = 5.0
    
    # Iterate over tour edges to apply specific penalties
    # This is O(n) where n is number of nodes, which is 100. This is fast.
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        
        # Check if this edge has high usage
        # We check the count in the symmetric matrix
        count = edge_n_used[u, v]
        
        if count > median_usage:
            # Apply heavy penalty to both directions
            perturbed[u, v] *= penalty_strength
            perturbed[v, u] *= penalty_strength
            
    # Optional: Apply a small global penalty to all tour edges to encourage slight variation
    # even if they aren't "high usage", to prevent getting stuck if median is low.
    # But the "Hard Tabu" on high-usage edges is the main driver.
    
    # Ensure symmetry (should be symmetric already if we update both, but just in case)
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure finiteness
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
