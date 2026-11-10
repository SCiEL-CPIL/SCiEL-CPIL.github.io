import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Construct a graph where edge weights reflect pair compatibility (inverse waste).
    2. Use the Fiedler vector of the graph Laplacian to obtain a 1D compatibility coordinate.
    3. Heuristic is the product of these coordinates, normalized.
    4. Fallback to direct waste-based heuristic if spectral method fails.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Compute pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Valid mask: pair sum <= capacity
    valid = pair_sums <= capacity
    
    # Initialize heuristic matrix with small positive value for invalid entries
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if not np.any(valid):
        return eta
    
    # For valid entries, compute waste
    waste = capacity - pair_sums
    
    # Direct waste-based values (used for graph construction and fallback)
    # Using the same sharp decay as the original program
    waste_valid = waste[valid]
    direct_values = 1.0 / (1.0 + waste_valid / 3.0) ** 3
    
    # --- Spectral Approach ---
    
    # 1. Build Graph Weight Matrix W
    W = np.zeros((n, n), dtype=float)
    W[valid] = direct_values
    
    # 2. Compute Laplacian L = D - W
    degree = np.sum(W, axis=1)
    L = np.diag(degree) - W
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to direct values
        eta[valid] = direct_values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Find the Fiedler vector (eigenvector of the smallest positive eigenvalue)
    tol = 1e-8
    pos_indices = np.where(eigenvalues > tol)[0]
    
    if len(pos_indices) == 0:
        # No positive eigenvalues? (e.g., disconnected graph with no edges between components)
        # Fallback to direct values
        eta[valid] = direct_values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta
        
    fiedler_idx = pos_indices[0]
    v = eigenvectors[:, fiedler_idx]
    
    # 3. Compute Heuristic from Fiedler vector
    # H_ij = v_i * v_j
    H = np.outer(v, v)
    
    # Normalize H so that the maximum positive value is 1.0
    # This preserves relative probabilities while ensuring a good scale
    max_h = np.max(H)
    if max_h > 0:
        H = H / max_h
    else:
        # Fallback if all values are non-positive
        eta[valid] = direct_values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta
        
    # Clip non-positive values to 1e-9
    H[H <= 0] = 1e-9
    
    # Ensure finite
    H[~np.isfinite(H)] = 1e-9
    
    # Check if the spectral heuristic is "degenerate" (e.g., too many 1e-9s)
    # If more than 50% of valid pairs are 1e-9, fallback to direct values
    # This is a safety measure in case the Fiedler vector is poorly distributed
    valid_count = np.sum(valid)
    small_count = np.sum(H[valid] <= 1e-8)
    
    if small_count > 0.5 * valid_count:
        eta[valid] = direct_values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta
    
    return H
