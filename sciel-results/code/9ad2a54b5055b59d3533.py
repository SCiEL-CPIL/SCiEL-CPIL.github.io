import numpy as np
import time
import scipy.spatial
import scipy.spatial.distance

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix (larger is more attractive) for ACO-TSP.
    
    Strategy:
    1. Reconstruct approximate 2D coordinates from the distance matrix using MDS.
    2. Compute the Delaunay triangulation of these points.
    3. Use the inverse distance as the base heuristic.
    4. Boost edges that are part of the Delaunay triangulation (natural neighbors).
       This encourages the formation of a planar, non-crossing tour, which is 
       geometrically favorable for TSP.
    """
    start_time = time.time()
    n = len(distance_matrix)
    
    # 1. Base heuristic: Inverse distance
    # Ensure diagonal is 0 to avoid division by zero, then set to 1e-9 later
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1.0) # Temporary to avoid div by 0 if any 0s off-diagonal? No, assume 0 on diag.
    
    # Actually, standard MDS needs 0 on diagonal.
    d_mds = distance_matrix.copy()
    np.fill_diagonal(d_mds, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = 1.0 / d_mds
    # Handle diagonal and any non-finite
    np.fill_diagonal(eta, 1e-9)
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # 2. Reconstruct coordinates using MDS
    # classical MDS
    try:
        # scipy.spatial.distance.sqform converts distance matrix to Gram matrix
        # D2 is squared distances
        D2 = d_mds ** 2
        # Centering matrix
        H = np.eye(n) - np.ones((n, n)) / n
        # Double centering
        B = -0.5 * H @ D2 @ H
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2 dimensions
        # Clip negative eigenvalues to 0
        eigvals = np.clip(eigvals, 0, None)
        coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
        
    except Exception:
        # Fallback to simple 1/d if MDS fails
        return eta
    
    # 3. Delaunay Triangulation
    try:
        tri = scipy.spatial.Delaunay(coords)
        # tri.simplices gives the triangles
        # We want to mark edges in the triangulation
        # Note: Delaunay triangulation of 2D points gives a planar graph.
        # Edges in this graph are "natural neighbors".
        
        # Create a boolean matrix for Delaunay edges
        delaunay_edges = np.zeros((n, n), dtype=bool)
        
        for simplex in tri.simplices:
            i, j, k = simplex
            delaunay_edges[i, j] = True
            delaunay_edges[j, i] = True
            delaunay_edges[i, k] = True
            delaunay_edges[k, i] = True
            delaunay_edges[j, k] = True
            delaunay_edges[k, j] = True
            
    except Exception:
        # If triangulation fails, just return base heuristic
        return eta
    
    # 4. Apply boost to Delaunay edges
    # Edges in the triangulation are more likely to be part of the optimal tour
    # because they connect nearby points without crossing.
    # We multiply by a factor > 1 for these edges.
    # Let's use a moderate boost, e.g., 1.5x
    boost_factor = 1.5
    eta[delaunay_edges] *= boost_factor
    
    # 5. Additional refinement: Convex Hull
    # Points on the convex hull must be visited in their cyclic order in the optimal tour.
    # We can identify the convex hull and boost the edges that connect consecutive hull points.
    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_vertices = hull.vertices
        # The vertices are in counter-clockwise order
        # Connect each vertex to the next
        for idx in range(len(hull_vertices)):
            u = hull_vertices[idx]
            v = hull_vertices[(idx + 1) % len(hull_vertices)]
            # Boost these specific edges further
            eta[u, v] *= 1.2
            eta[v, u] *= 1.2
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > 290:
        pass # Should be fast enough
    
    # Ensure finite and positive
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
