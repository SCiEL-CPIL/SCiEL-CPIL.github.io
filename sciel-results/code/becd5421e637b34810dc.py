"""Heuristic for ant colony TSP: Delaunay-boosted inverse distance with edge-length ratio scaling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    diag = np.eye(n, dtype=bool)

    # Recover 2D points from distance matrix using MDS (first 2 principal components)
    # This lets us compute Delaunay triangulation
    try:
        from scipy.spatial import Delaunay
        from scipy.spatial.distance import cdist
        from scipy.linalg import eigh
    except ImportError:
        # Fallback: just use inverse distance
        safe_D = D.copy()
        safe_D[diag] = 1.0
        H = 1.0 / safe_D
        H[diag] = 1e-9
        H = np.where(np.isfinite(H), H, 1e-9)
        H = np.maximum(H, 1e-9)
        return H

    # Double-centered Gram matrix for MDS
    J = np.eye(n) - np.ones((n, n)) / n
    # Convert squared distances to Gram: G = -0.5 * J * D^2 * J
    D2 = D ** 2
    G = -0.5 * J @ D2 @ J
    # Eigendecomposition
    eigvals, eigvecs = eigh(G)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top 2 components
    if eigvals[0] > 0:
        X = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    else:
        X = np.zeros((n, 2))

    # Compute Delaunay triangulation
    try:
        tri = Delaunay(X)
    except Exception:
        tri = None

    # Build Delaunay edge set
    delaunay_edges = set()
    if tri is not None:
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    delaunay_edges.add((min(a, b), max(a, b)))

    # Base heuristic: inverse distance
    safe_D = D.copy()
    safe_D[diag] = 1.0
    base = 1.0 / safe_D

    # Compute edge lengths for ratio scaling
    # For each pair (i,j), the ratio is D[i,j] / median_edge_length
    off_diag_D = D[~diag].ravel()
    median_len = float(np.median(off_diag_D))
    if median_len < 1e-15:
        median_len = 1.0

    # Build heuristic matrix
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 1e-9
                continue
            d = D[i, j]
            if d < 1e-15:
                H[i, j] = 1e-9
                continue
            # Base inverse distance
            h = 1.0 / d
            # Delaunay boost with edge-length ratio scaling
            key = (min(i, j), max(i, j))
            if key in delaunay_edges:
                # Shorter local edges get stronger boost
                ratio = d / median_len
                # Boost factor: stronger for shorter edges (ratio < 1)
                boost = 2.0 + 1.5 * (1.0 / max(ratio, 0.1))
                h *= boost
            else:
                # Smooth long-edge falloff for non-Delaunay edges
                ratio = d / median_len
                if ratio > 1.0:
                    # Penalize long non-Delaunay edges
                    falloff = 1.0 / (1.0 + 0.5 * (ratio - 1.0) ** 2)
                    h *= falloff
            H[i, j] = h

    # Ensure no zeros or non-finite values
    H[diag] = 1e-9
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)

    return H
# EVOLVE-BLOCK-END
