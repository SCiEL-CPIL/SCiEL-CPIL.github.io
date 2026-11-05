"""Rewritten program using spectral (Fiedler vector) approach for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Assign priority scores to vertices based on the Fiedler vector (eigenvector
    corresponding to the smallest positive eigenvalue of the graph Laplacian).
    Vertices with larger absolute Fiedler components are considered more peripheral
    (e.g., leaves in a tree-like structure), making them safer to include in an
    independent set without blocking many other vertices.
    
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
    
    # Build Laplacian L = D - A
    # For efficiency, work with sparse representation
    A_sparse = csr_matrix(adjacency)
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        if n <= 1:
            # Special case for very small graphs
            return -degrees.copy()
        
        # We want to find the eigenvector corresponding to the smallest *positive*
        # eigenvalue. For a connected graph, this is the 2nd smallest eigenvalue (Fiedler).
        # For a disconnected graph, there are multiple zero eigenvalues. We want the
        # first one that is strictly positive.
        
        # Fetch a few smallest eigenvalues to be safe.
        k = min(n, 5)
        if k < 2:
             # If n is very small, eigsh might struggle or return trivial results
             return -degrees.astype(float)

        # eigsh with sigma=0 finds eigenvalues closest to 0.
        # which='LM' means largest magnitude after shift, i.e., closest to 0.
        eigenvalues, eigenvectors = eigsh(L, k=k, sigma=0, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Find the first eigenvalue that is effectively positive
        # Using a small epsilon to handle numerical noise
        eps = 1e-10
        fiedler_idx = None
        for i in range(k):
            if eigenvalues[i] > eps:
                fiedler_idx = i
                break
        
        if fiedler_idx is None:
            # All eigenvalues are ~0, graph is a set of isolated nodes or empty
            # Fallback to degree (all 0, so arbitrary order, but let's use degree for stability)
            return -degrees.astype(float)
            
        fiedler_vec = eigenvectors[:, fiedler_idx]
        
        # Use absolute value as priority: larger |fiedler| -> higher priority.
        # This prioritizes vertices at the "ends" of the spectral structure (periphery).
        scores = np.abs(fiedler_vec)
        
        # Normalize scores to [0, 1] range for consistent tie-breaking scale
        if scores.max() > 0:
            scores = scores / scores.max()
            
        # Break ties by degree: lower degree gets slightly higher priority.
        # We want to add a small negative penalty for high degree.
        # Since scores are in [0,1], we can subtract a small fraction of normalized degree.
        max_deg = degrees.max()
        if max_deg > 0:
            degree_penalty = (degrees / max_deg) * 0.1
        else:
            degree_penalty = np.zeros(n)
            
        scores = scores - degree_penalty
        
        return scores.astype(float)
        
    except Exception:
        # Fallback to degree-based heuristic if spectral method fails
        return -degrees.astype(float)
# EVOLVE-BLOCK-END
