"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger values are more attractive.
    
    Strategy: Use a spectral clustering approach where we construct a graph with
    edge weights inversely proportional to residual space, then use the Fiedler
    vector to assign compatibility coordinates. The heuristic is based on how well
    items fit together, with strong preference for tight fits (low waste).
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Create the pair matrix: sum of each pair
    pair = demands[:, None] + demands[None, :]
    
    # Residual space when packing i and j together
    residual = capacity - pair
    
    # Only valid if both items fit together (residual >= 0)
    fits = residual >= 0
    
    # For fitting pairs, compute a weight that strongly favors tight fits.
    # We use a squared inverse waste approach: 1/(waste + epsilon)^2
    # where waste = residual. This gives very high weight to near-perfect fits.
    epsilon = 1e-6
    waste = np.maximum(residual, epsilon)
    
    # Weight: high for tight fits (small waste), low for loose fits (large waste)
    weight = np.where(fits, 1.0 / (waste ** 2), 0.0)
    
    # Normalize weights to [0, 1] range for numerical stability
    max_weight = weight.max() if weight.max() > 0 else 1.0
    weight = weight / max_weight
    
    # Now use spectral approach: construct a graph Laplacian from the weight matrix
    # and use the Fiedler vector to get a 1D compatibility coordinate.
    
    # Build adjacency matrix (symmetric)
    W = weight.copy()
    
    # Compute degree matrix
    degree = W.sum(axis=1)
    degree = np.maximum(degree, 1e-10)  # avoid division by zero
    
    # Compute normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    D_inv_sqrt = 1.0 / np.sqrt(degree)
    L = np.eye(n) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])
    
    # Ensure L is symmetric
    L = (L + L.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    # We want the Fiedler vector (second smallest eigenvalue)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # The smallest eigenvalue should be ~0 with eigenvector proportional to D^(1/2)
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Fiedler vector is the second one (index 1)
        fiedler = eigenvectors[:, 1]
        
        # Ensure non-negative: flip if necessary
        if fiedler.sum() < 0:
            fiedler = -fiedler
        
        # Normalize to [0, 1]
        f_min = fiedler.min()
        f_max = fiedler.max()
        if f_max - f_min > 1e-10:
            fiedler = (fiedler - f_min) / (f_max - f_min)
        else:
            fiedler = np.ones(n) / n
        
    except Exception:
        # Fallback if eigenvalue computation fails
        fiedler = demands / (demands.max() + 1e-10)
    
    # The heuristic: combine the direct fit weight with the spectral coordinate.
    # Items with similar coordinates (both high or both low) are more compatible.
    # We use: heuristic[i,j] = weight[i,j] * (1 - |fiedler[i] - fiedler[j]|)
    # This rewards both tight fits and spectral compatibility.
    
    coord_diff = np.abs(fiedler[:, None] - fiedler[None, :])
    spectral_compat = 1.0 - coord_diff
    
    # Combine: primary signal is the tight fit weight, modulated by spectral compatibility
    heuristic = weight * spectral_compat
    
    # Ensure diagonal is 0 (don't place item after itself)
    np.fill_diagonal(heuristic, 0.0)
    
    # Entries that are not positive become 1e-9
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure no non-finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
