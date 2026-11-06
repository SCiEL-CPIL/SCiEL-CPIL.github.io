"""Rewritten program using spectral (Fiedler vector) approach for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Assign priority scores to vertices based on the Fiedler vector (second smallest
    eigenvector of the graph Laplacian). Vertices with smaller absolute Fiedler
    component are considered more peripheral/less central, making them safer to
    include in an independent set.
    
    Returns an array of scores where higher score = higher priority (selected first).
    """
    n = adjacency.shape[0]
    
    # Convert to numpy array if needed
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure boolean/integer adjacency
    adjacency = adjacency.astype(float)
    
    # Handle edge case: empty graph
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute degree vector
    degrees = adjacency.sum(axis=1)
    
    # For n=200, use dense eigendecomposition for robustness
    if n <= 500:
        # Build Laplacian L = D - A densely
        D = np.diag(degrees)
        L = D - adjacency
        
        # Symmetrize to handle any numerical asymmetry
        L = (L + L.T) / 2.0
        
        # Full eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalue (eigh returns in ascending order already, but be safe)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # The Fiedler vector is the second one (index 1)
        fiedler_vec = eigenvectors[:, 1]
    else:
        # Fallback to sparse for larger graphs
        A_sparse = csr_matrix(adjacency)
        D_sparse = csr_matrix(np.diag(degrees))
        L = D_sparse - A_sparse
        
        try:
            eigenvalues, eigenvectors = eigsh(L, k=2, sigma=0, which='LM')
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            fiedler_vec = eigenvectors[:, 1]
        except Exception:
            return -degrees.astype(float)
    
    # Core idea: vertices with small |fiedler| are more peripheral
    # But we also want to break ties using structural information
    
    # Normalize Fiedler vector to have zero mean and unit variance for consistent scaling
    fiedler_mean = np.mean(fiedler_vec)
    fiedler_std = np.std(fiedler_vec)
    if fiedler_std > 1e-10:
        fiedler_norm = (fiedler_vec - fiedler_mean) / fiedler_std
    else:
        fiedler_norm = fiedler_vec.copy()
    
    # Primary score: negative absolute normalized Fiedler value
    # Smaller |fiedler| -> higher priority
    scores = -np.abs(fiedler_norm)
    
    # Secondary tie-breaker: prefer lower-degree vertices
    # Scale degree to [0, 1] range
    max_deg = degrees.max()
    if max_deg > 0:
        degree_norm = degrees / max_deg
    else:
        degree_norm = np.zeros_like(degrees)
    
    # Add a small penalty for high degree (they have more conflicts)
    # Weight: 0.3 so it's a tie-breaker but not dominant
    scores = scores - 0.3 * degree_norm
    
    # Tertiary: prefer vertices whose neighbors also have high |fiedler|
    # (i.e., picking this vertex removes "central" neighbors from contention)
    # Compute for each vertex: mean |fiedler| of its neighbors
    neighbor_fiedler_abs = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adjacency[i] > 0.5)[0]
        if len(neighbors) > 0:
            neighbor_fiedler_abs[i] = np.mean(np.abs(fiedler_vec[neighbors]))
        else:
            neighbor_fiedler_abs[i] = 0.0
    
    # Normalize neighbor_fiedler_abs
    nf_max = neighbor_fiedler_abs.max()
    if nf_max > 1e-10:
        neighbor_fiedler_norm = neighbor_fiedler_abs / nf_max
    else:
        neighbor_fiedler_norm = np.zeros_like(neighbor_fiedler_abs)
    
    # Small bonus for vertices that "cover" central neighbors
    scores = scores + 0.1 * neighbor_fiedler_norm
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
