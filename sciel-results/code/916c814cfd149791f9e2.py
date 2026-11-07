"""Rewritten program: Delaunay-based geometric heuristic for ACO-TSP."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import Delaunay

_DEADLINE_SECONDS = 300.0  # safety margin before 330s kill
_START_TIME = time.monotonic()


def _circumradius(a, b, c):
    """Circumradius of triangle with vertices a, b, c (2-D points)."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    # Side lengths
    a_len = np.hypot(bx - cx, by - cy)
    b_len = np.hypot(ax - cx, ay - cy)
    c_len = np.hypot(ax - bx, ay - by)
    # Area via cross product
    area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    if area < 1e-15:
        return 1e9  # degenerate
    return (a_len * b_len * c_len) / (4.0 * area)


def heuristics(distance_matrix):
    n = distance_matrix.shape[0]

    # Reconstruct 2-D points from distance matrix via MDS (simple embedding)
    # For uniform random points in unit square, we can use a lightweight approach:
    # Use the distance matrix to recover approximate coordinates.
    # Since we have exactly the pairwise distances of 50 points in 2D,
    # classical MDS (double centering) gives us the coordinates up to rotation/translation.
    D2 = distance_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen-decompose to get 2D coordinates
    eigvals, eigvecs = np.linalg.eigh(B)
    # Take top 2 eigenvalues (sorted descending)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Clamp small negative eigenvalues to 0
    eigvals = np.maximum(eigvals, 0.0)
    points = eigvecs[:, :2] * np.sqrt(eigvals[:2])

    # Build Delaunay triangulation
    try:
        tri = Delaunay(points)
        simplices = tri.simplices  # (num_triangles, 3)
    except Exception:
        # Fallback: just use inverse distance
        return 1.0 / (distance_matrix + 1e-9)

    # For each edge (i, j), collect circumradii of triangles containing that edge
    # edge_circumradii[i, j] = list of circumradii
    edge_circumradii = {}
    for tri_idx in range(simplices.shape[0]):
        i, j, k = simplices[tri_idx]
        R = _circumradius(points[i], points[j], points[k])
        for pair in [(i, j), (j, k), (k, i)]:
            edge_circumradii.setdefault(pair, []).append(R)

    # Build heuristic matrix
    heuristic = np.zeros((n, n))
    eps = 1e-9

    # Base: inverse distance
    base = 1.0 / (distance_matrix + eps)

    # Geometry factor: for each edge, use min circumradius (tightest triangle)
    # ratio = d / R_circ ; smaller ratio => better fit
    # We want attractance to increase when ratio is small.
    # Use: geom_factor = 1 / (1 + d / R_circ)  -- in (0, 1]
    # Actually, let's use: geom_factor = R_circ / (R_circ + d)  -- in (0, 0.5]
    # This rewards edges where the circumradius is large relative to the edge.

    # Precompute geometry factor for each edge
    geom = np.ones((n, n))
    for (i, j), radii_list in edge_circumradii.items():
        # Use the minimum circumradius (most constraining triangle)
        R_min = min(radii_list)
        d_ij = distance_matrix[i, j]
        # geom factor: R_min / (R_min + d_ij)
        # This is in (0, 0.5], with 0.5 when R_min >> d_ij
        # We want to BOOST edges with larger geom factor
        # So use geom = 1 + geom_factor to keep it > 1
        geom_val = R_min / (R_min + d_ij + eps)
        geom[i, j] = 1.0 + geom_val
        geom[j, i] = 1.0 + geom_val

    # Nearest-neighbor boost: edges in the k-NN graph get extra attractance
    k = 3
    nn_boost = np.ones((n, n))
    for i in range(n):
        # Exclude self
        d_i = distance_matrix[i].copy()
        d_i[i] = np.inf
        # Get k nearest neighbors
        nn_indices = np.argsort(d_i)[:k]
        for j in nn_indices:
            nn_boost[i, j] = 1.3
            nn_boost[j, i] = 1.3

    # Combine: heuristic = base * geom * nn_boost
    heuristic = base * geom * nn_boost

    # Zero the diagonal
    np.fill_diagonal(heuristic, 0.0)

    # Ensure all entries are positive (per task: non-positive becomes 1e-9)
    heuristic[heuristic <= 0] = 1e-9

    # Check finiteness
    if not np.all(np.isfinite(heuristic)):
        # Fallback
        fallback = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(fallback, 0.0)
        fallback[fallback <= 0] = 1e-9
        return fallback

    return heuristic
# EVOLVE-BLOCK-END
