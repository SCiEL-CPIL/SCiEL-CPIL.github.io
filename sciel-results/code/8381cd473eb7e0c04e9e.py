import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    Uses multiplicative penalties with saturation to penalize frequently used edges.
    Includes a 'topological disruption' component that heavily penalizes the most
    frequently used edges in the current tour to force structural change.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 330s limit
    
    n = edge_distance.shape[0]
    tour = np.asarray(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()
    
    # 1. Calculate saturated usage counts
    # This ensures penalty grows with usage but saturates, preventing numerical instability
    sat_usage = edge_n_used / (1.0 + edge_n_used)
    
    # 2. Initialize penalty matrix
    # We use a multiplicative penalty model: new_dist = dist * (1 + penalty)
    penalty = np.zeros_like(edge_distance, dtype=float)
    
    # 3. Global Penalty: Repel from all historically used edges
    # Alpha_global controls how strongly we repel from the "cloud" of previous solutions
    alpha_global = 0.5
    penalty += alpha_global * sat_usage
    
    # 4. Local Penalty: Extra repulsion from the current tour
    # Edges in the current tour are the ones maintaining the local optimum.
    # We penalize them more to encourage 2-opt to break them.
    alpha_local = 0.5
    
    # 5. Topological Disruption: Identify the most "stale" edges in the current tour
    # We find the k edges in the current tour with the highest usage counts.
    # We apply a significantly larger penalty to these specific edges.
    # This forces the tour to break its most entrenched structures.
    k_disrupt = 10
    alpha_disrupt = 2.0
    
    # Get tour edges
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        # Only consider unique edges (u < v) to avoid double counting in selection, 
        # but we will apply penalty to both (u,v) and (v,u)
        if u > v:
            u, v = v, u
        usage = edge_n_used[u, v]
        tour_edges.append((usage, u, v))
    
    # Sort by usage descending to find the most used edges
    tour_edges.sort(key=lambda x: x[0], reverse=True)
    
    # Apply local and disruption penalties
    for idx in range(n):
        usage, u, v = tour_edges[idx]
        
        # Base local penalty for all current tour edges
        p_local = alpha_local * sat_usage[u, v]
        
        # Disruption penalty for the top-k most used edges
        if idx < k_disrupt:
            p_local += alpha_disrupt * sat_usage[u, v]
            
        penalty[u, v] += p_local
        penalty[v, u] += p_local
        
        # Time check
        if time.time() > deadline:
            break

    # 6. Apply multiplicative penalty
    # Ensure the factor is positive (it is, as penalties are non-negative)
    modified_distance = edge_distance * (1.0 + penalty)
    
    # 7. Ensure symmetry (numerical safety)
    modified_distance = (modified_distance + modified_distance.T) / 2.0
    
    # 8. Ensure finite values
    # If any values are inf or nan, replace them. 
    # Max distance in unit square is sqrt(2) ~ 1.414. 
    # A large penalty might push it high, but not inf unless penalty is inf.
    # sat_usage is bounded by 1, so penalty is bounded.
    modified_distance = np.nan_to_num(modified_distance, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Final safety check: if we are too close to deadline, return a simpler version?
    # The loop above is O(N log N) which is very fast for N=100.
    # The main cost is the matrix operations.
    
    return modified_distance
