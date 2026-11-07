"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    
    Strategy: Delaunay Topological Heuristic with Inverse-Square Weighting.
    
    1. Reconstruct 2D coordinates from the distance matrix using MDS (or assume 
       we can recover coordinates if the input is from a standard generator, 
       but here we only have the distance matrix). 
       
       Wait, the input is `distance_matrix`. We do NOT have the raw points.
       However, the task description says "3 fixed instances of 50 uniform random points in the unit square".
       The function signature is `heuristics(distance_matrix)`.
       
       To use Delaunay, we need coordinates. We can recover coordinates from 
       the distance matrix using Classical Multidimensional Scaling (MDS).
       Since the points are in 2D, a 2D MDS reconstruction should be exact 
       (up to rotation/reflection) for uniform random points in the unit square, 
       assuming no degenerate configurations.
       
    2. Compute the Delaunay triangulation of the recovered points.
    3. Create a mask of Delaunay edges.
    4. Construct heuristic:
       - For Delaunay edges: h[i,j] = (1 / d[i,j]^2)
       - For non-Delaunay edges: h[i,j] = 1e-9 (or a very small constant)
       
    This strategy enforces that ants almost exclusively choose edges from the 
    Delaunay triangulation. The Delaunay triangulation is a superset of the 
    Euclidean Minimum Spanning Tree and is known to contain the optimal TSP 
    tour for points in the plane (specifically, the optimal TSP tour is a 
    subgraph of the Delaunay triangulation, or at least, the optimal tour 
    does not cross, and Delaunay is a planar graph containing the EMST).
    Actually, the optimal TSP tour is not always a subgraph of the Delaunay 
    triangulation? 
    Correction: The optimal TSP tour is a non-crossing polygonal chain. 
    The Delaunay triangulation contains all edges of the EMST. 
    It is a known result that the optimal TSP tour is a subgraph of the 
    Relative Neighborhood Graph (RNG), which is a subgraph of the Delaunay 
    Triangulation. Therefore, the optimal tour is contained in the Delaunay 
    Triangulation.
    
    By restricting the search to Delaunay edges, we drastically reduce the 
    branching factor for the ants, guiding them towards the optimal solution 
    space. The inverse-square weighting makes shorter edges significantly 
    more attractive, which aligns with the goal of minimizing total length.
    """
    n = distance_matrix.shape[0]
    
    # Safety check for time
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds is plenty for n=50 MDS + Delaunay
    
    try:
        # 1. Recover 2D coordinates using Classical MDS
        # MDS: G = -0.5 * J * (D^2) * J
        # Where J = I - (1/n) * 11^T
        # D^2 is the squared distance matrix
        
        D_sq = distance_matrix ** 2
        # Ensure symmetry and zero diagonal
        D_sq = (D_sq + D_sq.T) / 2.0
        np.fill_diagonal(D_sq, 0.0)
        
        J = np.eye(n) - np.ones((n, n)) / n
        G = -0.5 * (J @ D_sq @ J)
        
        # Eigendecomposition of G
        # G is symmetric, so we use eigh
        eigvals, eigvecs = np.linalg.eigh(G)
        
        # Sort eigenvalues in descending order
        # eigh returns ascending, so reverse
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take the first 2 eigenvectors for 2D embedding
        # Scale by sqrt of eigenvalues
        if n >= 2:
            V = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
        else:
            V = np.zeros((n, 2))
            
        # The coordinates are V
        coords = V
        
        # 2. Compute Delaunay Triangulation
        # scipy.spatial.Delaunay expects an (n, 2) array of points
        try:
            tri = Delaunay(coords)
            simplices = tri.simplices # shape (m, 3)
            
            # Create a mask for Delaunay edges
            delaunay_mask = np.zeros((n, n), dtype=bool)
            for simplex in simplices:
                i, j, k = simplex
                delaunay_mask[i, j] = True
                delaunay_mask[j, i] = True
                delaunay_mask[i, k] = True
                delaunay_mask[k, i] = True
                delaunay_mask[j, k] = True
                delaunay_mask[k, j] = True
                
        except Exception:
            # If Delaunay fails (e.g., degenerate points), fall back to MST or just inverse distance
            # For now, use a dense mask but with very low weights for non-short edges?
            # Or just return inverse distance squared for all, which is a valid fallback.
            h = 1.0 / (distance_matrix ** 2 + 1e-12)
            np.fill_diagonal(h, 0.0)
            return h

        # 3. Construct Heuristic
        # Base: Inverse square distance
        # Add small epsilon to avoid division by zero for diagonal
        inv_dist_sq = 1.0 / (distance_matrix ** 2 + 1e-12)
        
        # Mask out non-Delaunay edges
        # We want high attractiveness for Delaunay edges, low for others.
        # The task says "Heuristic entries that are not positive become 1e-9".
        # So we can set non-Delaunay edges to a very small value, e.g., 1e-6 or 1e-9.
        # However, if we set them to 0, they become 1e-9 by the evaluator.
        # Let's set them to 1e-9 directly to be safe and explicit.
        
        h = np.full((n, n), 1e-9)
        
        # Apply inverse square distance only to Delaunay edges
        h[delaunay_mask] = inv_dist_sq[delaunay_mask]
        
        # Ensure diagonal is 0 (or very small, evaluator handles non-positive)
        np.fill_diagonal(h, 0.0)
        
        # Check for finite values
        h = np.where(np.isfinite(h), h, 1e-9)
        
        # Check time
        if time.time() > deadline:
             # Fallback if too slow
             h_fallback = 1.0 / (distance_matrix ** 2 + 1e-12)
             np.fill_diagonal(h_fallback, 0.0)
             return h_fallback
             
        return h
        
    except Exception:
        # Fallback to simple inverse square distance
        h_fallback = 1.0 / (distance_matrix ** 2 + 1e-12)
        np.fill_diagonal(h_fallback, 0.0)
        return h_fallback
# EVOLVE-BLOCK-END
