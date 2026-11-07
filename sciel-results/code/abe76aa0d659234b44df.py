# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def _compute_delaunay_heuristic(distance_matrix, coordinates):
    """
    Computes a heuristic matrix based on Delaunay triangulation edges.
    Edges in the Delaunay triangulation are assigned high weights.
    Other edges are assigned weights based on inverse distance, but penalized
    to be much less attractive than Delaunay edges.
    """
    n = distance_matrix.shape[0]
    
    # Initialize heuristic with inverse distance (standard ACO heuristic)
    # We handle the diagonal (self-loops) by setting a large distance
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, 1e6)
    base_heuristic = 1.0 / dist_safe
    
    # Create Delaunay triangulation
    # coordinates shape is (n, 2)
    try:
        tri = Delaunay(coordinates)
    except Exception:
        # Fallback if triangulation fails (e.g., collinear points, though unlikely in uniform square)
        # Just return base heuristic
        return base_heuristic
    
    # Identify edges in the triangulation
    # tri.simplices gives triangles (indices of 3 points)
    # We need to extract unique edges from these triangles
    edges = set()
    for simplex in tri.simplices:
        i, j, k = simplex
        edges.add((i, j))
        edges.add((j, i))
        edges.add((j, k))
        edges.add((k, j))
        edges.add((i, k))
        edges.add((k, i))
    
    # Create a boolean mask for Delaunay edges
    delaunay_mask = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        delaunay_mask[i, j] = True
    
    # Construct the final heuristic matrix
    # Strategy: 
    # 1. Delaunay edges get a large boost.
    # 2. Non-Delaunay edges are heavily penalized.
    # 
    # The ACO probability is proportional to tau * eta.
    # If we set eta_high for Delaunay and eta_low for others, the ACO will strongly prefer
    # paths that follow the triangulation. This restricts the search to "planar" paths,
    # avoiding crossings which are always suboptimal in Euclidean TSP/CVRP.
    
    # We want the ratio between Delaunay and non-Delaunay to be significant but not infinite.
    # Let's scale the base heuristic such that the maximum is 1.0 (approx, since min dist is small).
    # Actually, base_heuristic is already 1/dist.
    # Let's apply a multiplicative factor.
    
    # Factor for Delaunay edges: 10.0
    # Factor for non-Delaunay edges: 0.1 (or just keep base, but let's dampen it to encourage planarity)
    # Actually, if we dampen non-Delaunay too much, we might disconnect the graph or force
    # long detours. But Delaunay triangulation is connected and dense enough (3n-6 edges).
    # For n=50, ~144 edges. Total possible ~2450. So ~6% of edges are allowed "full" access.
    # This is a strong constraint.
    
    # Let's try:
    # H[i,j] = base_heuristic[i,j] * (10.0 if delaunay_mask[i,j] else 0.5)
    # This makes Delaunay edges 20x more attractive than non-Delaunay edges of same length.
    
    heuristic_matrix = base_heuristic.copy()
    heuristic_matrix[~delaunay_mask] *= 0.5
    heuristic_matrix[delaunay_mask] *= 10.0
    
    # Ensure positivity and finiteness
    heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
    
    return heuristic_matrix

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a Delaunay Triangulation strategy.
    
    1. Compute Delaunay triangulation of the customer coordinates (including depot).
    2. Create a heuristic matrix where edges in the triangulation are highly attractive.
    3. Edges not in the triangulation are penalized to discourage crossing paths.
    4. This guides the ACO towards planar solutions, which are known to be optimal 
       or near-optimal for Euclidean routing problems.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 290.0

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # Check time budget
    if time.time() > deadline - 10:
        # Fallback to simple inverse distance if we are running out of time
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        fallback = 1.0 / dist_safe
        fallback = np.where(fallback > 0, fallback, 1e-9)
        return fallback

    # Compute Delaunay-based heuristic
    heuristic_matrix = _compute_delaunay_heuristic(distance_matrix, coordinates)
    
    # Final validation
    if not np.all(np.isfinite(heuristic_matrix)):
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        heuristic_matrix = 1.0 / dist_safe
        heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
        
    if heuristic_matrix.shape != distance_matrix.shape:
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        heuristic_matrix = 1.0 / dist_safe
        heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)

    return heuristic_matrix
# EVOLVE-BLOCK-END
