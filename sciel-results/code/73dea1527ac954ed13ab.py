"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.float64)
    
    # Spectral Method for Maximum Cut
    # We want to maximize the cut size, which is equivalent to minimizing x^T A x
    # for x in {-1, 1}^n.
    # The Rayleigh quotient is minimized by the eigenvector corresponding to the smallest eigenvalue.
    
    try:
        # Compute eigenvalues and eigenvectors
        # For a symmetric matrix, we use eigh
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
        
        # The smallest eigenvalue corresponds to the first column in the returned matrix
        # (eigh returns eigenvalues in ascending order)
        min_eigenvector = eigenvectors[:, 0]
        
        # Convert to 0/1 labels based on sign
        # Positive -> 1, Negative -> 0
        # If component is 0, assign 0 (arbitrary)
        labels = (min_eigenvector > 0).astype(np.int8)
        
    except Exception:
        # Fallback to random if spectral fails (unlikely)
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        
    return labels
# EVOLVE-BLOCK-END
