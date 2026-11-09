"""Guided local search penalty for TSP: adaptive additive penalty with spectral smoothing."""
import numpy as np
import time

# Cache for spectral decomposition of the distance matrix structure
_spectral_cache = {}


def _get_spectral_basis(edge_distance):
    """
    Compute a low-dimensional spectral basis for the distance matrix.
    Uses the top few eigenvectors of the graph Laplacian of the distance matrix.
    Returns a matrix S of shape (n, k) where k is small (e.g., 5).
    """
    n = edge_distance.shape[0]
    # Use a hash of the first few entries to identify the matrix structure
    # Since we get fresh copies each time, we use a simple key based on n and a checksum
    # For efficiency, we'll just compute it each time but keep k small
    # Actually, let's just compute the eigenvectors of the normalized Laplacian
    
    # Build adjacency from distance: small distance = strong connection
    # Use inverse distance as weight (with safety for zero diagonal)
    W = edge_distance.copy()
    np.fill_diagonal(W, 0.0)
    # Use a kernel to make it positive definite-ish
    sigma = np.median(W[W > 0]) if np.any(W > 0) else 1.0
    if sigma <= 0:
        sigma = 1.0
    W = np.exp(-W**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)
    
    # Degree matrix
    d = W.sum(axis=1)
    d[d == 0] = 1.0
    D_inv_sqrt = 1.0 / np.sqrt(d)
    
    # Normalized Laplacian
    L = np.eye(n) - D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :]
    
    # Get smallest eigenvalues (excluding trivial ones)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Take the first few non-trivial eigenvectors
        # Skip the first (constant) one
        k = min(5, n - 1)
        S = eigvecs[:, 1:1+k]
        return S
    except Exception:
        # Fallback: random orthonormal
        rng = np.random.RandomState(42)
        S = rng.randn(n, min(5, n-1))
        Q, _ = np.linalg.qr(S)
        return Q


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation:
    1. Additive penalty proportional to edge usage history
    2. Spectral smoothing to create global topological bias
    """
    n = edge_distance.shape[0]
    
    # Base true distances
    base = edge_distance.copy()
    
    # --- Component 1: Additive usage penalty ---
    # Scale penalty by mean distance to be adaptive
    mean_dist = np.mean(base)
    if mean_dist <= 0:
        mean_dist = 1.0
    
    # Adaptive penalty weight: stronger when usage counts are high
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    # Smooth scaling: penalty grows sublinearly with usage
    penalty_scale = 0.12 * mean_dist * np.sqrt(1.0 + max_used)
    
    # Additive penalty on used edges
    penalized = base + penalty_scale * edge_n_used
    
    # --- Component 2: Spectral topological bias ---
    # Get spectral basis
    S = _get_spectral_basis(base)
    k = S.shape[1]
    
    # Project edge usage into spectral space to get a global bias
    # The idea: edges that connect nodes with similar spectral coordinates
    # are "smooth" and should be less penalized; edges that jump in spectral
    # space are "oscillatory" and should be penalized more.
    
    # Spectral distance between nodes: ||S[i] - S[j]||^2
    # This is a global measure of how "far apart" nodes are in the low-dim embedding
    spectral_dist = np.zeros((n, n), dtype=np.float64)
    # Efficient computation: ||S_i - S_j||^2 = ||S_i||^2 + ||S_j||^2 - 2 S_i·S_j
    S_sq = (S ** 2).sum(axis=1)  # (n,)
    S_dot = S @ S.T  # (n, n)
    spectral_dist = S_sq[:, None] + S_sq[None, :] - 2 * S_dot
    np.maximum(spectral_dist, 0.0, out=spectral_dist)
    
    # Normalize spectral distance
    spec_max = spectral_dist.max()
    if spec_max > 0:
        spectral_dist /= spec_max
    
    # Weight the spectral bias: penalize edges that are spectrally far apart
    # This creates a global structure that discourages crossing patterns
    spec_weight = 0.05 * mean_dist
    penalized += spec_weight * spectral_dist * edge_n_used
    
    # Ensure finite and non-negative
    penalized = np.where(np.isfinite(penalized), penalized, base)
    penalized = np.maximum(penalized, 0.0)
    
    return penalized
