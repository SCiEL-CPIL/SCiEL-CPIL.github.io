"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Combines three structural signals:
    1. Inverse-squared distance (tight local clustering)
    2. Angular continuity around the depot (sector-based, non-crossing routes)
    3. Spectral embedding via Fiedler vector of the distance-weighted Laplacian
       (enforces global space-filling, non-crossing structure)
    """
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    eps = 1e-9
    
    # 1. Base Heuristic: Inverse Squared Distance
    base_heuristic = 1.0 / (dm + eps)**2
    
    # 2. Angular Continuity Heuristic
    depot_coords = coords[0]
    dx = coords[:, 0] - depot_coords[0]
    dy = coords[:, 1] - depot_coords[1]
    angles = np.arctan2(dy, dx)
    
    angle_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
    angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
    
    lambda_angle = 4.0
    angular_factor = np.exp(-lambda_angle * (angle_diff ** 2))
    
    # 3. Spectral embedding via Fiedler vector
    # Build weighted adjacency from inverse-squared distances (excluding self-loops)
    W = base_heuristic.copy()
    np.fill_diagonal(W, 0.0)
    
    # Graph Laplacian: L = D - W
    degree = W.sum(axis=1)
    L = np.diag(degree) - W
    
    # Compute eigenvalues and eigenvectors
    # For n=51, full eigendecomposition is fast
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigvecs columns are sorted by ascending eigenvalue from eigh
        fiedler = eigvecs[:, 1]
    except Exception:
        # Fallback: use x-coordinate as spectral proxy
        fiedler = coords[:, 0]
    
    # Normalize Fiedler vector to zero mean, unit variance
    fiedler_mean = fiedler.mean()
    fiedler_std = fiedler.std()
    if fiedler_std > 1e-12:
        fiedler_norm = (fiedler - fiedler_mean) / fiedler_std
    else:
        fiedler_norm = fiedler - fiedler_mean
    
    # Spectral similarity: cosine-like similarity of embedded positions
    # For 1D embedding, similarity is exp(-alpha * (fiedler_i - fiedler_j)^2)
    fiedler_diff = fiedler_norm[:, np.newaxis] - fiedler_norm[np.newaxis, :]
    lambda_spectral = 3.0
    spectral_factor = np.exp(-lambda_spectral * (fiedler_diff ** 2))
    
    # 4. Combine all factors
    heuristic = base_heuristic.copy()
    
    # Apply angular and spectral factors only to customer-customer edges
    mask = np.ones((n, n), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False
    
    heuristic[mask] *= angular_factor[mask] * spectral_factor[mask]
    
    # 5. Demand Awareness
    demand_penalty = 1.0 / (1.0 + 0.05 * dem)
    heuristic *= demand_penalty[np.newaxis, :]
    
    # 6. Sanitize
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
