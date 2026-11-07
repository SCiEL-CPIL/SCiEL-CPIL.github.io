import numpy as np
from scipy.spatial import Delaunay
import time

def heuristics(distance_matrix):
    """
    Computes a heuristic matrix that biases toward Delaunay edges with inverse-square
    distance, and uses a softer inverse distance for non-Delaunay edges.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s kill
    
    d = np.asarray(distance_matrix, dtype=float)
    
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
        
    n = d.shape[0]
    d_work = d.copy()
    
    # Set diagonal to large value
    np.fill_diagonal(d_work, 1e18)
    
    # Avoid division by zero
    min_d = 1e-9
    d_work = np.where(d_work > min_d, d_work, min_d)
    
    # Try to recover coordinates using classical MDS
    coords = None
    try:
        D2 = d ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))[:, None]
    except Exception:
        coords = None
    
    if coords is not None and time.time() < deadline:
        try:
            tri = Delaunay(coords)
            del_edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        del_edges.add((min(a,b), max(a,b)))
            
            # Base heuristic: 1/d^1 for all edges (softer)
            alpha = 1.0
            h = 1.0 / np.power(d_work, alpha)
            
            # Boost Delaunay edges with inverse square (steeper)
            beta = 2.0
            for (i, j) in del_edges:
                h[i, j] = 1.0 / np.power(d_work[i, j], beta)
                h[j, i] = 1.0 / np.power(d_work[j, i], beta)
                
        except Exception:
            beta = 2.0
            h = 1.0 / np.power(d_work, beta)
    else:
        beta = 2.0
        h = 1.0 / np.power(d_work, beta)
    
    # Handle diagonal
    np.fill_diagonal(h, 0.0)
    
    # Normalize to mean 1 over off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    valid_h = h[mask]
    if len(valid_h) > 0:
        mean_h = np.mean(valid_h)
        if mean_h > 0:
            h = h / mean_h
        else:
            h = np.ones_like(h)
    else:
        h = np.ones_like(h)
        
    # Ensure non-finite values are handled
    h = np.where(np.isfinite(h), h, 0.0)
    
    # Entries that are not positive become 1e-9
    h = np.where(h > 0, h, 1e-9)
    
    return h
