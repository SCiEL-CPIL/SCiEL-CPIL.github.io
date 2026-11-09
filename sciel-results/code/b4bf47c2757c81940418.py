"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, QhullError
import scipy.spatial as sp

def _recover_coords_2d(D):
    """Recover 2D coordinates from a Euclidean distance matrix using classical MDS."""
    n = D.shape[0]
    sq = D ** 2
    ones = np.ones((n, n))
    J = np.eye(n) - ones / n
    B = -0.5 * J @ sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.maximum(eigvals[:2], 0)
    coords = eigvecs[:, :2] * np.sqrt(eigvals)
    return coords

def _compute_voronoi_neighbors_and_areas(coords):
    """
    Compute Voronoi diagram, return a set of neighbor pairs and cell areas.
    """
    n = len(coords)
    try:
        vor = Voronoi(coords)
    except QhullError:
        return set(), np.ones(n)
    
    neighbors = set()
    # vor.ridge_points contains pairs of input points that share a Voronoi edge
    # vor.ridge_vertices contains the vertices of the ridge
    for p1, p2 in vor.ridge_points:
        if p1 != -1 and p2 != -1:
            neighbors.add((min(p1, p2), max(p1, p2)))
    
    # Calculate approximate cell areas
    # vor.regions contains indices of vertices for each region
    # vor.vertices contains the coordinates of the vertices
    # Note: Unbounded regions have -1 in their region list. We will approximate area
    # by clipping to the bounding box or just using 0 for unbounded and averaging for bounded.
    # A simpler robust metric: distance to nearest neighbor as a proxy for cell size?
    # Or just use the area of the convex hull of the region if bounded.
    
    areas = np.zeros(n)
    for i in range(n):
        region_idx = vor.regions[i]
        if len(region_idx) == 0 or -1 in region_idx:
            # Unbounded, approximate with average bounded area or a default
            areas[i] = 1.0 # Placeholder, will be normalized later
        else:
            verts = vor.vertices[region_idx]
            # Shoelace formula for polygon area
            x = verts[:, 0]
            y = verts[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas[i] = area
            
    # Normalize areas to have mean 1 for scaling
    if np.mean(areas) > 0:
        areas = areas / np.mean(areas)
    else:
        areas = np.ones(n)
        
    return neighbors, areas

def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ACO on TSP using Voronoi topology.
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    # Base heuristic: inverse distance
    H = 1.0 / (distance_matrix + 1e-9)
    
    try:
        coords = _recover_coords_2d(distance_matrix)
        neighbors, areas = _compute_voronoi_neighbors_and_areas(coords)
        
        # Create a boolean matrix for Voronoi neighbors
        is_neighbor = np.zeros((n, n), dtype=bool)
        for u, v in neighbors:
            is_neighbor[u, v] = True
            is_neighbor[v, u] = True
            
        # Boost neighbor edges significantly
        # Non-neighbor edges get a penalty based on distance relative to local scale
        # Local scale for edge (i,j) can be sqrt(area_i * area_j)
        
        # Calculate local scale matrix
        # S_ij = sqrt(area_i * area_j)
        scale_matrix = np.sqrt(np.outer(areas, areas))
        
        # Heuristic adjustment
        # For neighbors: H * Boost
        # For non-neighbors: H * Penalty(d / scale)
        
        boost_factor = 25.0
        
        # Penalty function: 
        # If d is small relative to scale, it's a "short" non-neighbor edge (maybe missed by Voronoi due to numerical issues or just close).
        # If d is large relative to scale, it's a "long" edge.
        # We want to suppress long edges.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_dist = distance_matrix / (scale_matrix + 1e-9)
            
        # Penalty is 1 for small relative dist, drops off for large
        # Using a Gaussian-like or power-law falloff
        # penalty = exp(-relative_dist^2) or 1 / (1 + relative_dist^2)
        # Let's use 1 / (1 + relative_dist^2)
        
        penalty = 1.0 / (1.0 + relative_dist**2)
        
        # Apply boost to neighbors, penalty to others
        # H_final = H * (is_neighbor * boost_factor + (1 - is_neighbor) * penalty)
        # But we want to ensure neighbors are very attractive.
        
        adjustment = np.where(is_neighbor, boost_factor, penalty)
        
        H = H * adjustment
        
    except Exception:
        # Fallback to simple inverse distance with global median penalty
        med = np.median(distance_matrix[~np.eye(n, dtype=bool)])
        if med > 0:
            relative_dist = distance_matrix / med
            penalty = 1.0 / (1.0 + relative_dist**2)
            H = H * penalty

    # Ensure positive values
    H = np.maximum(H, 1e-9)
    
    # Symmetrize
    H = (H + H.T) / 2
    
    return H
# EVOLVE-BLOCK-END
