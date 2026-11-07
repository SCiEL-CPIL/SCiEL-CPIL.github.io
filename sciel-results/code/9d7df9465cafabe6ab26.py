import numpy as np
import time

_start_time = None

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Safety check for timeout
    if time.time() - _start_time > 300:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    d_out = edge_distance.copy()
    
    # Mask out diagonal
    np.fill_diagonal(d_out, 0.0)

    # 1. Identify the set of edges in the current local optimum tour
    tour_edges = set()
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                if u < v:
                    tour_edges.add((u, v))
                else:
                    tour_edges.add((v, u))
    except Exception:
        return edge_distance.copy()

    # 2. Identify "new" edges (never used in any previous local optimum)
    # edge_n_used is a symmetric count matrix.
    # An edge is "new" if edge_n_used[i, j] == 0 and it's not a self-loop.
    
    # We want to encourage the 2-opt to consider these edges.
    # Strategy: Assign a very low cost to a subset of these new edges.
    # To avoid creating a trivially short tour of just new edges (which might be geometrically impossible or disconnected),
    # we limit the number of "bait" edges.
    
    # Get all new edges
    # For efficiency, we can just look at the upper triangle
    # Since n=100, n^2 is 10000, which is small.
    
    new_edges = []
    # Iterate through upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            if edge_n_used[i, j] == 0:
                new_edges.append((i, j))

    # We want to pick a small number of these new edges to "force" into the solution.
    # Let's pick 2-3 edges to create a "break" in the current tour structure.
    # Randomly selecting might be bad if they are far apart.
    # But since we don't have coordinates, we rely on the 2-opt to fix the geometry.
    # Actually, the 2-opt will minimize the sum of distances. If we make some edges very cheap,
    # the 2-opt might include them if they help reduce the total length in the perturbed space.
    
    # A better strategy for GLS:
    # Instead of just penalizing used edges, we should REWARD new edges.
    # But the interface returns a distance matrix.
    # If we reduce the distance of new edges, 2-opt will prefer them.
    
    # Let's select a few new edges to reduce.
    # How many? Maybe 1 or 2.
    # Let's try to select new edges that are "close" to the current tour in some sense?
    # We don't have coordinates.
    
    # Alternative: Just pick a random new edge and make it very cheap.
    # This forces a specific change.
    
    if new_edges:
        # Select 1 random new edge to "force"
        # Using a deterministic pseudo-random based on round to avoid global state issues if not needed
        # But we can just use np.random if we seed it? No, let's keep it deterministic or simple.
        # Let's pick the first one? No, that's biased.
        # Let's use the current time or a counter?
        # Actually, just picking one is enough to perturb.
        
        # Let's pick 1 edge.
        # To make it more effective, let's pick an edge that is NOT in the current tour (guaranteed by definition)
        # and assign it a distance that is slightly less than the minimum distance in the matrix?
        # Or just 0?
        # If we set it to 0, 2-opt will definitely try to use it if it can fit it in.
        
        # Let's set the cost of 1 new edge to 0.0.
        # This creates a "free" edge. The 2-opt will try to incorporate it.
        # This is a strong perturbation.
        
        # Pick a random index
        # Since we don't want global state, let's just pick the first one for simplicity, 
        # or better, use a hash of the current tour to pick a consistent one?
        # Let's just pick one at index 0 of the new_edges list.
        # Wait, new_edges order depends on loop.
        # Let's pick one based on the sum of the tour to make it vary?
        
        # Actually, the most robust "different" strategy is:
        # Penalize the current tour edges heavily (as before) BUT ALSO
        # Identify the "most expensive" edge in the current tour and try to replace it?
        
        # Let's stick to the "Force New Edge" strategy.
        # It is structurally different from "Penalize Old Edges".
        # "Penalize Old" pushes away. "Force New" pulls towards unexplored.
        
        # Let's select 1 new edge.
        # To avoid always picking the same one (e.g., (0,1) if it's always new),
        # we can rotate through them or use a simple hash.
        # For now, let's just pick one.
        
        # Let's pick the new edge with the smallest true distance?
        # That would be the most "natural" new edge to add.
        
        # Find new edge with min distance
        min_dist = np.inf
        best_new_edge = None
        for i, j in new_edges:
            if edge_distance[i, j] < min_dist:
                min_dist = edge_distance[i, j]
                best_new_edge = (i, j)
                
        if best_new_edge is not None:
            u, v = best_new_edge
            # Set distance to 0 to strongly encourage its inclusion
            d_out[u, v] = 0.0
            d_out[v, u] = 0.0

    # Additionally, apply a mild penalty to the current tour edges to discourage staying there
    # This combines "Pull" (new edge) with "Push" (old edges)
    for u, v in tour_edges:
        # Increase distance of current tour edges
        # Factor 1.5 is mild
        d_out[u, v] *= 1.5
        d_out[v, u] *= 1.5

    # Safety check
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()

    return d_out
