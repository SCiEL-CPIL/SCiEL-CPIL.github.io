"""Spectral clustering approach for ant colony heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix using spectral clustering on a compatibility graph.
    
    Strategy: Build a graph where edge weights are inversely proportional to
    residual space (waste) for fitting pairs. Use the Fiedler vector of the
    graph Laplacian to assign a 1D compatibility coordinate to each item.
    The heuristic is the product of these coordinates for fitting pairs.
    """
    start_time = time.time()
    deadline = start_time + 270  # Safety margin before 300s limit
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Precompute
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    waste = capacity - sums  # positive for fitting pairs
    
    # Build adjacency matrix: edge weight = 1/(waste + eps) for fitting pairs
    epsilon = 1e-3
    W = np.zeros((n, n), dtype=np.float64)
    W[fits] = 1.0 / (waste[fits] + epsilon)
    # Zero out diagonal
    np.fill_diagonal(W, 0.0)
    
    # Compute graph Laplacian: L = D - W
    degree = W.sum(axis=1)
    D = np.diag(degree)
    L = D - W
    
    # Compute eigenvalues and eigenvectors of L
    # We want the Fiedler vector (second smallest eigenvalue)
    try:
        evals, evecs = np.linalg.eigh(L)
        # Sort by eigenvalue (eigh already returns sorted)
        # Fiedler vector is the eigenvector for the second smallest eigenvalue
        fiedler = evecs[:, 1]
    except Exception:
        # Fallback: use item sizes as coordinates
        fiedler = d.copy()
    
    # Normalize Fiedler vector to have zero mean and unit variance
    fiedler_mean = fiedler.mean()
    fiedler_std = fiedler.std()
    if fiedler_std > 1e-12:
        fiedler = (fiedler - fiedler_mean) / fiedler_std
    else:
        fiedler = d.copy()
    
    # The heuristic: H[i,j] = fiedler[i] * fiedler[j] for fitting pairs
    # But we want larger values to be more attractive, so we need to be careful
    # about the sign. Let's use the absolute value or square to ensure positivity.
    # Actually, let's use the product but ensure it's positive by using abs or square.
    # Better: use (fiedler[i] + offset) * (fiedler[j] + offset) where offset shifts to positive
    
    # Alternative: use the magnitude of the Fiedler vector as a "compatibility score"
    # and multiply by the inverse waste term to directly reward tight fits
    compat = np.abs(fiedler)
    
    # Combine with inverse waste: H[i,j] = compat[i] * compat[j] * (1/(waste[i,j] + eps))
    # This rewards both high compatibility and tight fits
    raw_H = np.outer(compat, compat) * W
    
    # Apply fit mask: non-fitting pairs get 1e-9
    H = np.where(fits, raw_H, 1e-9)
    
    # Ensure finiteness
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Safety check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
    
    return H
# EVOLVE-BLOCK-END
