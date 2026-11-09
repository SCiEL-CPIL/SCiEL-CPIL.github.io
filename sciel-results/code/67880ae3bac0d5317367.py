import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    perturbed = edge_distance.copy()
    
    # Normalize usage counts
    max_used = edge_n_used.max()
    if max_used == 0:
        usage_norm = np.zeros_like(edge_n_used)
    else:
        usage_norm = edge_n_used / max_used
        
    # Calculate penalties for each node based on local geometry
    # and add to incident edges
    curvature_penalties = np.zeros((n, n))
    
    tour = local_opt_tour
    for i in range(n):
        u = tour[(i - 1) % n]
        v = tour[(i + 1) % n]
        
        d_ui = edge_distance[u, i]
        d_iv = edge_distance[i, v]
        d_uv = edge_distance[u, v]
        
        # Avoid division by zero
        denom = d_ui + d_iv
        if denom > 1e-9:
            # S is 1 for straight, ~0 for U-turn
            S = d_uv / denom
            # K is 0 for straight, ~1 for U-turn
            K = 1.0 - S
            # Clamp K to [0, 1] just in case of numerical issues or non-Euclidean noise
            K = np.clip(K, 0.0, 1.0)
            
            # Add penalty to incident edges
            curvature_penalties[u, i] += K
            curvature_penalties[i, v] += K
            # Symmetry
            curvature_penalties[i, u] += K
            curvature_penalties[v, i] += K
        else:
            K = 0.0
            
    # Combine penalties
    # We penalize edges in the current tour.
    # The penalty is a mix of curvature and usage.
    # Let's apply it multiplicatively.
    
    alpha_curv = 0.8
    alpha_use = 0.5
    
    # We only want to penalize edges that are actually in the tour?
    # The task says "add the current tour's edges to edge_n_used".
    # Standard GLS penalizes edges in the local optimum.
    # We should only modify the edges in the current tour to avoid distorting the rest of the matrix unnecessarily,
    # though distorting non-tour edges can also guide 2-opt.
    # However, 2-opt swaps two edges. If we penalize non-tour edges, we might discourage valid swaps.
    # It is safer to only penalize the edges currently in the tour.
    
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        
        # Get the combined penalty factor for this edge
        # Usage part
        use_pen = alpha_use * usage_norm[u, v]
        
        # Curvature part: average curvature of the two endpoints?
        # Or just use the max?
        # Let's use the average curvature score of the two nodes incident to this edge.
        # Actually, the curvature calculated at node i affects edges (u,i) and (i,v).
        # So edge (u,v) in the tour is incident to u and v.
        # The curvature penalty accumulated at u and v in `curvature_penalties` is relevant.
        # But `curvature_penalties` was built by adding K_i to edges incident to i.
        # So `curvature_penalties[u, v]` contains the sum of K_u (from u's step) and K_v (from v's step)?
        # No, in the loop:
        #   Step for node i: adds K_i to (u,i) and (i,v).
        #   So edge (u,v) is NOT directly penalized by K_i.
        #   Edge (u,v) is penalized by K_u (when i=u, v is next) and K_v (when i=v, u is prev).
        #   So `curvature_penalties[u, v]` holds K_u + K_v.
        
        curv_pen = alpha_curv * curvature_penalties[u, v]
        
        total_pen = use_pen + curv_pen
        
        # Apply multiplicatively
        factor = 1.0 + total_pen
        perturbed[u, v] *= factor
        perturbed[v, u] *= factor
        
    return perturbed
