"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.optimize import minimize


def _recover_coords_2d(D):
    """Recover 2D coordinates from a Euclidean distance matrix using classical MDS."""
    n = D.shape[0]
    # Center the squared distance matrix
    sq = D ** 2
    # Double-centering: B = -1/2 * J * sq * J
    ones = np.ones((n, n))
    J = np.eye(n) - ones / n
    B = -0.5 * J @ sq @ J
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top 2 eigenvalues (should be non-negative for Euclidean distances)
    eigvals = np.maximum(eigvals[:2], 0)
    coords = eigvecs[:, :2] * np.sqrt(eigvals)
    return coords


def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ACO on TSP.
    
    Approach: Delaunay-based topological heuristic with adaptive boost
    - Recover 2D coordinates via classical MDS from the distance matrix.
    - Compute Delaunay triangulation to identify natural short, non-crossing edges.
    - Boost Delaunay edges with an adaptive factor based on edge length relative
      to the median Delaunay edge length.
    - Apply a smooth penalty for very long edges that are not Delaunay.
    - Combine with inverse distance base.
    
    Returns: matrix of same shape as distance_matrix, larger = more attractive
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    # Base heuristic: inverse distance
    H = 1.0 / (distance_matrix + 1e-9)
    
    # Try to recover coordinates and compute Delaunay
    try:
        coords = _recover_coords_2d(distance_matrix)
        tri = Delaunay(coords)
        
        # Extract edges from Delaunay triangulation
        is_delaunay = np.zeros((n, n), dtype=bool)
        delaunay_lengths = []
        for simplex in tri.simplices:
            u, v, w = simplex
            is_delaunay[u, v] = True
            is_delaunay[v, u] = True
            is_delaunay[v, w] = True
            is_delaunay[w, v] = True
            is_delaunay[w, u] = True
            is_delaunay[u, w] = True
            # Record edge lengths for adaptive boost
            delaunay_lengths.append(distance_matrix[u, v])
            delaunay_lengths.append(distance_matrix[v, w])
            delaunay_lengths.append(distance_matrix[w, u])
        
        delaunay_lengths = np.array(delaunay_lengths)
        median_dl = np.median(delaunay_lengths) if len(delaunay_lengths) > 0 else 1.0
        
        # Adaptive boost: stronger boost for shorter Delaunay edges
        # Boost factor = base_boost * (median_dl / edge_length)^alpha, capped
        alpha = 0.5
        base_boost = 15.0
        max_boost = 100.0
        
        # Compute boost for Delaunay edges
        dl_boost = np.ones((n, n))
        if median_dl > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = median_dl / (distance_matrix + 1e-9)
                boost_vals = base_boost * (ratio ** alpha)
            boost_vals = np.clip(boost_vals, 1.0, max_boost)
            dl_boost = np.where(is_delaunay, boost_vals, 1.0)
        
        H *= dl_boost
        
        # Penalty for very long non-Delaunay edges
        # Smooth penalty: edges much longer than median get reduced attractiveness
        long_threshold = 3.0 * median_dl
        penalty = np.ones((n, n))
        long_mask = (~is_delaunay) & (distance_matrix > long_threshold)
        # Smooth falloff: penalty = 1 / (1 + (d/threshold)^2) for long edges
        with np.errstate(divide='ignore', invalid='ignore'):
            falloff = 1.0 / (1.0 + (distance_matrix / long_threshold) ** 2)
        penalty = np.where(long_mask, falloff, 1.0)
        
        H *= penalty
        
    except (QhullError, Exception):
        # Fallback: just use inverse distance with mild long-edge penalty
        med = np.median(distance_matrix[~np.eye(n, dtype=bool)])
        if med > 0:
            long_mask = distance_matrix > 3.0 * med
            with np.errstate(divide='ignore', invalid='ignore'):
                falloff = 1.0 / (1.0 + (distance_matrix / (3.0 * med)) ** 2)
            H *= np.where(long_mask, falloff, 1.0)
    
    # Ensure positive values
    H = np.maximum(H, 1e-9)
    
    # Symmetrize
    H = (H + H.T) / 2
    
    return H
# EVOLVE-BLOCK-END
