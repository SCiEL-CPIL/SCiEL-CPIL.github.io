# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix based on Delaunay triangulation topology.
    Only edges in the Delaunay triangulation get high attractiveness;
    all other edges get very low attractiveness.
    """
    n = distance_matrix.shape[0]
    
    # Reconstruct points from the distance matrix using classical MDS
    D = distance_matrix
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 eigenvalues for 2D embedding
    eigenvalues[0:2] = np.maximum(eigenvalues[0:2], 0)
    
    # Compute 2D coordinates
    points = eigenvectors[:, 0:2] * np.sqrt(eigenvalues[0:2])
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(points)
        simplices = tri.simplices
    except Exception:
        # Fallback: use k-nearest neighbors as a proxy for local structure
        k = 6
        heuristic = np.full((n, n), 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            k_nearest = np.argpartition(dists, k)[:k]
            for j in k_nearest:
                d = distance_matrix[i, j]
                attractiveness = 1.0 / (d + 1e-9)
                heuristic[i, j] = max(heuristic[i, j], attractiveness)
                heuristic[j, i] = max(heuristic[j, i], attractiveness)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # Build set of Delaunay edges
    delaunay_edges = set()
    for simplex in simplices:
        i, j, k = simplex
        delaunay_edges.add((min(i, j), max(i, j)))
        delaunay_edges.add((min(j, k), max(j, k)))
        delaunay_edges.add((min(i, k), max(i, k)))
    
    # Initialize heuristic with very low base value for non-Delaunay edges
    low_value = 1e-6
    
    heuristic = np.full((n, n), low_value)
    np.fill_diagonal(heuristic, 0.0)
    
    # For each Delaunay edge, assign high attractiveness based on inverse distance
    for (i, j) in delaunay_edges:
        d = distance_matrix[i, j]
        attractiveness = 1.0 / (d + 1e-9)
        heuristic[i, j] = attractiveness
        heuristic[j, i] = attractiveness
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
