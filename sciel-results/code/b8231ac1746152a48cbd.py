"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Set diagonal to large value
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Sparsify: Keep only top K nearest neighbors for each node
    K = 10
    eta = np.full_like(d_safe, 1e-9)
    
    for i in range(n):
        # Get indices of K smallest distances
        indices = np.argsort(d_safe[i])[:K]
        for j in indices:
            eta[i, j] = 1.0 / (d_safe[i, j] ** 2)
            
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Compute spectral gap penalty using Fiedler vector
    # Build a weighted graph Laplacian from inverse distances
    # Use only the K nearest neighbor edges to keep it sparse
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        indices = np.argsort(d_safe[i])[:K]
        for j in indices:
            W[i, j] = 1.0 / d_safe[i, j]
    
    # Compute degree matrix
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    
    # Compute eigenvalues and eigenvectors
    try:
        evals, evecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(evals)
        fiedler = evecs[:, idx[1]]
        
        # Apply spectral penalty: dampen edges where Fiedler coordinates differ significantly
        # This discourages long-range "bottleneck" connections
        fiedler_diff = np.abs(fiedler[:, None] - fiedler[None, :])
        # Normalize fiedler_diff to [0, 1]
        max_diff = fiedler_diff.max()
        if max_diff > 0:
            fiedler_diff = fiedler_diff / max_diff
        
        # Multiplicative penalty: reduce attractiveness for edges with large Fiedler differences
        # Use a smooth penalty function
        penalty = 1.0 - 0.5 * fiedler_diff  # Range [0.5, 1.0]
        
        # Apply penalty only to non-zero heuristic entries
        mask = eta > 1e-8
        eta[mask] *= penalty[mask]
        
    except Exception:
        pass  # If spectral computation fails, return the distance-based heuristic
    
    # Ensure finite and positive after penalty
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
