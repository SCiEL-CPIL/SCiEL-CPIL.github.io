import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task using spectral clustering.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate waste
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix
    H = np.zeros((n, n), dtype=np.float64)
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    if not np.any(fits):
        # No pairs fit, return minimal heuristic
        H[:] = 1e-9
        return H
    
    # Build graph for spectral clustering
    # Edge weight: inversely proportional to residual space if two items are packed together
    # For fitting pairs, weight = 1 / (waste + eps) to favor tight fits
    eps = 1e-6
    W = np.zeros((n, n), dtype=np.float64)
    W[fits] = 1.0 / (waste[fits] + eps)
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    
    # Build Laplacian: L = D - W
    D = np.diag(degrees)
    L = D - W
    
    # Find Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
    # Use scipy if available, otherwise numpy
    try:
        from scipy.linalg import eigh
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, 1])
        fiedler = eigenvectors[:, 0]
    except ImportError:
        # Fallback to numpy
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Fiedler vector is the second smallest eigenvalue's eigenvector
        fiedler = eigenvectors[:, 1]
    
    # Ensure fiedler vector is real
    fiedler = np.real(fiedler)
    
    # Normalize the Fiedler vector to have zero mean and unit variance
    fiedler = fiedler - np.mean(fiedler)
    std = np.std(fiedler)
    if std > 1e-10:
        fiedler = fiedler / std
    
    # Set heuristic matrix entries to the product of these coordinates
    # H[i, j] = fiedler[i] * fiedler[j] for fitting pairs
    H[fits] = np.outer(fiedler, fiedler)[fits]
    
    # Ensure all values are finite and positive
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline - 10:
        return H
    
    return H
