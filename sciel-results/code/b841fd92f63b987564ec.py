import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Hard Exclusion Perturbation for Guided Local Search.
    
    Strategy:
    Instead of applying additive penalties which might be too weak to break strong local optima
    or too strong to create a valid path, we completely exclude the edges of the current
    local-optimal tour from being selected by the subsequent 2-opt pass.
    
    By setting the distance of all edges in the current tour to a very large value (effectively
    infinity, but finite to satisfy constraints), we force the 2-opt algorithm to find a
    completely different set of edges for those positions. This guarantees a significant
    structural change to the tour, breaking out of the local basin much more aggressively
    than gradual penalty accumulation.
    
    The 'edge_n_used' is ignored for the penalty calculation itself, as the exclusion is
    binary: either the edge is in the current tour (and thus excluded) or it is not.
    This simplifies the logic and removes the need for tuning penalty coefficients.
    """
    
    # Ensure we have a copy to modify
    modified_dist = edge_distance.copy()
    
    n = len(local_opt_tour)
    if n < 2:
        return modified_dist
    
    # Determine a large value to act as "infinity"
    # The max distance in the unit square is sqrt(2) ~ 1.414.
    # A value of 1000.0 is safely large enough to prevent selection in 2-opt
    # while remaining a finite float.
    LARGE_VAL = 1000.0
    
    # Identify all edges in the current tour and set their distances to LARGE_VAL
    # The tour is a sequence of nodes. Edges are (tour[i], tour[(i+1)%n])
    # We must handle the closure (last node to first node).
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Set both directions in the symmetric matrix
        modified_dist[u, v] = LARGE_VAL
        modified_dist[v, u] = LARGE_VAL
        
    return modified_dist
