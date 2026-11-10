import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # 1. Construct Weight Matrix
    # Residual R_ij = C - (d_i + d_j)
    sums = demands[:, None] + demands[None, :]
    residuals = capacity - sums
    
    # Mask for fitting pairs
    fit_mask = residuals >= 0
    
    # Calculate weights
    # Use exponential decay of residual to avoid division by zero and smoothness
    # W_ij = exp(-R_ij / capacity)
    # If R is 0, W=1. If R is C, W=exp(-1) ~ 0.37.
    # This rewards tight fits.
    W = np.where(fit_mask, np.exp(-residuals / capacity), 0.0)
    
    # Symmetric, diagonal is 0
    np.fill_diagonal(W, 0)
    
    # 2. Spectral Decomposition
    # We want the principal eigenvector (largest eigenvalue) of W
    # This represents the "centrality" or "connectivity" of each item
    # in the graph of fitting pairs.
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    # The largest eigenvalue is the last one
    principal_vec = eigenvectors[:, -1]
    
    # Ensure positive coordinates (eigenvector can be negated arbitrarily)
    # The principal eigenvector of a positive matrix is positive (Perron-Frobenius)
    # But just in case of numerical issues or disconnected components, take abs
    coords = np.abs(principal_vec)
    
    # Normalize coordinates to [0, 1] to keep heuristic values bounded
    max_coord = np.max(coords)
    if max_coord > 0:
        coords = coords / max_coord
    else:
        coords = np.ones(n)
        
    # 3. Heuristic Matrix
    # H_ij = coords_i * coords_j
    H = np.outer(coords, coords)
    
    # Apply fitting mask: if not fitting, set to 1e-9
    H = np.where(fit_mask, H, 1e-9)
    
    # Ensure positive
    H = np.where(H > 0, H, 1e-9)
    
    return H
