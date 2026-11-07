import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Generate a perturbation matrix using a continuous repulsive force field model.
    
    Strategy:
    Instead of just penalizing the edges in the current tour, we calculate a penalty
    for every edge (u, v) based on how "close" it is to the current tour's edges.
    We define "closeness" in the metric space induced by the tour structure.
    
    Specifically, for each edge (u, v) not in the current tour, we look at the 
    minimum "detour cost" to insert (u, v) into the current tour or replace an 
    existing edge. However, a simpler and computationally efficient proxy for 
    "distance from the tour" in the context of 2-opt is the difference between 
    the edge length and the average edge length of the current tour, weighted by 
    a decay factor based on how many times similar edges (geometrically close) 
    have been used in the past (edge_n_used).
    
    A more direct "force field" approach:
    1. Identify the set of edges in the current local_opt_tour.
    2. For every pair (i, j), calculate a penalty that is high if (i, j) is in 
       the tour, and decays for edges that are "near" the tour in terms of 
       potential 2-opt swaps.
       
    Let's refine the "force field" idea:
    The current program penalizes only the edges in the tour. This is a "hard" 
    constraint. A "soft" force field would penalize edges that are *likely* to 
    be part of the local optimum's "basin of attraction".
    
    Heuristic for the force field:
    Let T be the set of edges in local_opt_tour.
    For any edge (i, j):
    - If (i, j) in T: Apply a strong penalty proportional to 1 + alpha * edge_n_used[i, j].
    - If (i, j) not in T: Apply a smaller penalty if (i, j) is geometrically 
      close to any edge in T. "Geometrically close" can be approximated by 
      comparing the distance d(i, j) to the distances of edges in T.
      
    Actually, a more robust structural change is to view the problem as 
    minimizing the overlap with the *history* of tours, not just the current one.
    The "force field" is centered on the *accumulated* usage.
    
    New Model:
    Penalty(i, j) = Base_Penalty * exp(-k * (d(i, j) - d_min)) 
    where d_min is the minimum distance among the edges in the current tour?
    No, that doesn't make sense for repulsion.
    
    Let's try a different structural framing: 
    **Simulated Annealing-inspired Cooling of the Penalty.**
    The current program uses random penalties. 
    The new strategy will use a deterministic, annealed penalty schedule 
    combined with a "memory" of edge usage.
    
    However, the directive asks for a "repulsive force field centered at the 
    current tour edges... decays with the Euclidean distance to the nearest 
    point on the tour in the metric space".
    
    Implementation of the Force Field:
    1. Extract the list of edges in `local_opt_tour`.
    2. For each edge (i, j) in the full matrix:
       - If (i, j) is in the tour, it is the "center" of a repulsive force.
       - The penalty applied to (i, j) itself is high.
       - The penalty applied to *other* edges (k, l) is based on how similar 
         (k, l) is to (i, j). 
         
    A simpler, effective "force field" for TSP 2-opt perturbation:
    Penalize edges based on their "regret" relative to the current tour.
    But let's stick to the prompt's specific inspiration: 
    "penalty as a repulsive force field centered at the current tour edges, 
    where the magnitude decays with the Euclidean distance to the nearest 
    point on the tour in the metric space".
    
    This implies we should penalize edges (u, v) based on the distance from 
    the *midpoint* of (u, v) to the *midpoints* of the tour edges? 
    Or simply, penalize edges that are "close" in distance value to the tour edges?
    
    Let's interpret "distance to the nearest point on the tour in the metric space" 
    as: For an edge (u, v), find the tour edge (a, b) that minimizes 
    |d(u, v) - d(a, b)|. The penalty decays as this difference increases.
    This creates a "band" of high penalty around the distance values present 
    in the current tour. This discourages 2-opt from swapping in edges that 
    have similar lengths to the current tour's edges, effectively pushing 
    the tour towards edges with different length distributions, which often 
    correspond to different geometric structures.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    n = len(local_opt_tour)
    perturbed = edge_distance.copy()
    
    # 1. Extract tour edges and their distances
    tour_edges = []
    tour_dists = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v:
            u, v = v, u
        tour_edges.append((u, v))
        tour_dists.append(edge_distance[u, v])
        
    tour_dists = np.array(tour_dists)
    min_tour_dist = np.min(tour_dists)
    max_tour_dist = np.max(tour_dists)
    mean_tour_dist = np.mean(tour_dists)
    
    # 2. Define the force field parameters
    # We want to penalize edges that are "close" to the current tour's edge lengths.
    # This is a proxy for "structural similarity".
    
    # The "force" is strongest for edges in the tour, and decays for others 
    # based on how much their distance differs from the tour's distance distribution.
    
    # Vectorized approach for speed:
    # Get all unique pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    all_edges = np.column_stack((i_idx, j_idx))
    all_dists = edge_distance[i_idx, j_idx]
    
    # Calculate the "distance" of each edge's length to the nearest tour edge length
    # This is O(N^2 * N_tour) which is 10000 * 100 = 1e6 operations, acceptable.
    
    # For each edge, find min |d(edge) - d(tour_edge)|
    # We can do this by broadcasting
    # diff = np.abs(all_dists[:, np.newaxis] - tour_dists[np.newaxis, :])
    # min_diff = np.min(diff, axis=1)
    
    # To save memory and time, we can bin the distances or just compute directly if n is small.
    # n=100, so 4950 edges. 4950 * 100 = 495,000 comparisons. Very fast.
    
    diff = np.abs(all_dists[:, np.newaxis] - tour_dists[np.newaxis, :])
    min_diff = np.min(diff, axis=1)
    
    # The penalty magnitude:
    # Base penalty for edges in the tour (min_diff == 0)
    # Decay for others.
    
    # We use an exponential decay: penalty = A * exp(-min_diff / sigma)
    # A should be large enough to break the local optimum.
    # sigma controls the width of the "exclusion zone".
    
    # We also incorporate edge_n_used to make the penalty adaptive.
    # Edges used frequently in the past should have a stronger repulsive field.
    
    # Get edge_n_used for all edges
    edge_n_used_vals = edge_n_used[i_idx, j_idx]
    
    # Adaptive sigma: if edges are used a lot, the field is sharper (smaller sigma)?
    # Or stronger?
    # Let's make the base amplitude A depend on the maximum usage count.
    max_usage = np.max(edge_n_used_vals) if edge_n_used_vals.size > 0 else 0
    
    # Normalize usage to [0, 1]
    if max_usage > 0:
        usage_norm = edge_n_used_vals / max_usage
    else:
        usage_norm = np.zeros_like(edge_n_used_vals)
    
    # Base penalty scale
    # Typical edge length ~0.3. We want penalty to be significant, e.g., 1.0 - 3.0.
    base_penalty = 2.0
    
    # Decay parameter
    sigma = 0.1 # Width of the exclusion zone in distance space
    
    # Calculate penalty for each edge
    # penalty = base_penalty * exp(-min_diff / sigma) * (1 + usage_factor * usage_norm)
    
    # We want to penalize the tour edges the most.
    # min_diff is 0 for tour edges.
    # So exp(0) = 1.
    
    # For non-tour edges, min_diff > 0, so penalty < base_penalty.
    
    # Let's add a random jitter to avoid deterministic cycles, but keep it small.
    jitter = np.random.uniform(0, 0.1, size=len(all_dists))
    
    penalties = base_penalty * np.exp(-min_diff / sigma) + jitter
    
    # Apply the penalty to the symmetric matrix
    # We need to map all_edges back to the matrix indices
    perturbed[i_idx, j_idx] += penalties
    perturbed[j_idx, i_idx] += penalties
    
    # Safety check
    if time.time() > deadline:
        return edge_distance
        
    # Ensure finiteness
    if not np.all(np.isfinite(perturbed)):
        return edge_distance
        
    return perturbed
