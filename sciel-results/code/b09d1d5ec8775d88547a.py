"""Heuristic matrix for ACO-TSP: Planar Graph Metric via Delaunay Triangulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0

def _compute_mds_coordinates(dist_matrix):
    """
    Compute 2D coordinates from distance matrix using Classical MDS.
    Returns None if it fails or takes too long.
    """
    n = dist_matrix.shape[0]
    if n < 4:
        return None
    
    # Check deadline
    if time.time() > _DEADLINE - 30.0:
        return None

    try:
        # Classical MDS
        # Step 1: Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (dist_matrix ** 2) @ J
        
        # Step 2: Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2
        # If eigenvalues are negative, MDS is not metric, but we can still use the vectors
        coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
        
        # Center and scale
        coords -= coords.mean(axis=0)
        # Scale to [0, 1] box roughly
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        ranges = max_coords - min_coords
        ranges[ranges == 0] = 1.0
        coords = (coords - min_coords) / ranges
        
        return coords
    except Exception:
        return None

def _build_delaunay_graph(coords, dist_matrix):
    """
    Build a planar graph (Delaunay triangulation) from coordinates.
    Returns a weighted adjacency matrix for the graph.
    """
    n = len(coords)
    if n < 3:
        # Fallback to complete graph for tiny n
        return dist_matrix.copy()
    
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(coords)
        
        # Initialize graph with zeros
        graph = np.zeros((n, n))
        
        # Add edges from triangulation
        # The vertices array contains triangles. We extract unique edges.
        # For n=50, this is small.
        vertices = tri.vertices
        for simplex in vertices:
            v1, v2, v3 = simplex
            # Add edges (v1, v2), (v2, v3), (v3, v1)
            for u, v in [(v1, v2), (v2, v3), (v3, v1)]:
                if graph[u, v] == 0:
                    graph[u, v] = dist_matrix[u, v]
                    graph[v, u] = dist_matrix[u, v]
        
        # Check connectivity. Delaunay triangulation of a point set is always connected
        # if the points are not collinear, but let's be safe.
        # If any node is isolated (shouldn't happen for n>=3 non-collinear), 
        # we might need to add edges. But for random points, it's fine.
        
        return graph
    except Exception:
        # Fallback: Use k-NN graph if Delaunay fails
        k = 4
        graph = np.zeros((n, n))
        for i in range(n):
            # Find k nearest neighbors
            dists = dist_matrix[i].copy()
            dists[i] = np.inf
            nn_indices = np.argsort(dists)[:k]
            for j in nn_indices:
                graph[i, j] = dist_matrix[i, j]
                graph[j, i] = dist_matrix[i, j]
        return graph

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix.
    
    Strategy:
    1. Compute MDS coordinates.
    2. Construct Delaunay triangulation (planar graph).
    3. Compute all-pairs shortest paths in this planar graph.
    4. Heuristic = 1 / (planar_distance^2).
       This naturally penalizes long-range jumps and edges that would "cross" 
       the planar structure, favoring the local connectivity of the triangulation.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 2:
        return np.ones((n, n))
    
    # Fallback heuristic if MDS or graph construction fails
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    np.fill_diagonal(base_heuristic, 0.0)
    
    coords = None
    if time.time() < _DEADLINE - 45.0:
        coords = _compute_mds_coordinates(d)
    
    if coords is None:
        # If MDS fails, just return the base heuristic (inverse distance squared)
        # This is better than crashing.
        base_heuristic = np.where(np.isfinite(base_heuristic) & (base_heuristic > 0), base_heuristic, 1e-9)
        np.fill_diagonal(base_heuristic, 0.0)
        return base_heuristic
    
    # Build Planar Graph
    graph = _build_delaunay_graph(coords, d)
    
    # Check if graph is connected. If not, fallback.
    # Simple check: are there any zeros in the upper triangle that shouldn't be?
    # Actually, let's just try to compute shortest paths. If it fails or returns inf, we handle it.
    
    try:
        # Compute all-pairs shortest paths
        # Using scipy.sparse.csgraph.shortest_path for efficiency
        from scipy.sparse.csgraph import shortest_path
        
        # Convert to sparse for shortest_path
        # Note: shortest_path expects a weight matrix. 
        # We treat the graph as undirected.
        # scipy.sparse.csgraph.shortest_path works on directed graphs by default.
        # We have a symmetric matrix.
        
        # Ensure no self loops for the path calculation (diagonal should be 0)
        # But the input to shortest_path usually ignores diagonal or treats it as 0.
        # Let's create a sparse matrix.
        from scipy.sparse import csr_matrix
        sparse_graph = csr_matrix(graph)
        
        # Compute shortest paths
        # directed=False
        planar_dists = shortest_path(sparse_graph, directed=False)
        
        # Check for disconnected components (inf values)
        if np.any(np.isinf(planar_dists)):
            # Fallback to base heuristic if graph is disconnected
            # (Should be rare for Delaunay, but possible with numerical issues)
            base_heuristic = 1.0 / (d + 1e-9) ** 2
            np.fill_diagonal(base_heuristic, 0.0)
            base_heuristic = np.where(np.isfinite(base_heuristic) & (base_heuristic > 0), base_heuristic, 1e-9)
            np.fill_diagonal(base_heuristic, 0.0)
            return base_heuristic
        
        # The diagonal will be 0. We need to handle division by zero.
        # We want heuristic[i, j] to be high for short planar distances.
        # However, we only care about edges that exist in the original complete graph.
        # The heuristic matrix must be of shape (n, n).
        
        # For pairs (i, j) where i != j:
        # If they are connected in the planar graph, use 1/planar_dist^2.
        # If not (shouldn't happen if connected), use base heuristic.
        
        # We want to ensure that edges NOT in the Delaunay triangulation still have
        # a reasonable heuristic value, but penalized by the "detour" cost.
        # The planar distance captures the "detour" through the planar graph.
        
        # Heuristic definition:
        # h[i, j] = 1 / (planar_dist[i, j] + epsilon)^2
        
        # This makes direct edges (which are in the graph) have low planar_dist (equal to Euclidean).
        # Edges that are not in the graph have higher planar_dist (sum of edges in the shortest path).
        # Thus, they are less attractive.
        
        epsilon = 1e-9
        final_heuristic = 1.0 / (planar_dists + epsilon) ** 2
        
        # Set diagonal to 0
        np.fill_diagonal(final_heuristic, 0.0)
        
        # Ensure finite and positive
        final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
        np.fill_diagonal(final_heuristic, 0.0)
        
        # Normalize to mean 1 for stability? 
        # The ACO algorithm multiplies pheromone * heuristic.
        # If the scale is off, it might affect convergence speed but not necessarily the final result 
        # if pheromone is also scaled or if the relative ratios are what matter.
        # However, standard practice often normalizes.
        # Let's normalize so the mean of non-zero elements is 1.
        mask = final_heuristic > 0
        if np.any(mask):
            mean_h = np.mean(final_heuristic[mask])
            if mean_h > 0:
                final_heuristic = final_heuristic / mean_h
        
        return final_heuristic
        
    except Exception:
        # Fallback to base heuristic
        base_heuristic = 1.0 / (d + 1e-9) ** 2
        np.fill_diagonal(base_heuristic, 0.0)
        base_heuristic = np.where(np.isfinite(base_heuristic) & (base_heuristic > 0), base_heuristic, 1e-9)
        np.fill_diagonal(base_heuristic, 0.0)
        return base_heuristic

# EVOLVE-BLOCK-END
