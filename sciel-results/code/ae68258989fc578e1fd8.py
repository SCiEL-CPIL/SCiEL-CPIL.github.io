"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices using a spectral approach.
    Uses the Fiedler vector (eigenvector of the second smallest eigenvalue
    of the graph Laplacian) combined with degree information.
    
    The idea: vertices with large |Fiedler component| are peripheral and
    less central to connectivity, making them safer picks for an independent
    set. We combine this with low-degree preference.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Compute degree
    degree = A.sum(axis=1)
    
    # Compute Laplacian: L = D - A
    D = np.diag(degree)
    L = D - A
    
    # For efficiency, use sparse eigenvalue solver if available
    # For n=200, dense eigendecomposition is fast enough
    try:
        # Get eigenvalues and eigenvectors of the Laplacian
        # We want the second smallest eigenvalue's eigenvector (Fiedler vector)
        # np.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # The smallest eigenvalue is 0 (for connected graphs) with eigenvector all ones
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigenvectors[:, 0] corresponds to eigenvalues[0] (smallest)
        # eigenvectors[:, 1] corresponds to eigenvalues[1] (second smallest)
        
        # Check if the graph is connected: if eigenvalues[1] is very small, 
        # the graph may be disconnected
        fiedler_vec = eigenvectors[:, 1]
        
        # Use magnitude of Fiedler vector as a measure of peripherality
        # Vertices with large |fiedler_vec[i]| are more peripheral
        fiedler_mag = np.abs(fiedler_vec)
        
        # Normalize to [0, 1]
        fmin, fmax = fiedler_mag.min(), fiedler_mag.max()
        if fmax > fmin + 1e-12:
            fiedler_norm = (fiedler_mag - fmin) / (fmax - fmin)
        else:
            fiedler_norm = np.zeros(n)
        
        # Combine: we want high priority for peripheral, low-degree vertices
        # Priority = alpha * (1 - degree_norm) + beta * fiedler_norm
        # Higher score = selected first
        
        # Normalize degree to [0, 1]
        dmin, dmax = degree.min(), degree.max()
        if dmax > dmin + 1e-12:
            degree_norm = (degree - dmin) / (dmax - dmin)
        else:
            degree_norm = np.zeros(n)
        
        # Weighting: emphasize peripherality and low degree
        # Empirically, a blend works well
        alpha = 0.5  # weight for low degree
        beta = 0.5   # weight for peripherality
        
        scores = alpha * (1.0 - degree_norm) + beta * fiedler_norm
        
        return scores.astype(np.float64)
        
    except Exception:
        # Fallback to degree-based if spectral fails
        return -degree.astype(np.float64)
# EVOLVE-BLOCK-END
