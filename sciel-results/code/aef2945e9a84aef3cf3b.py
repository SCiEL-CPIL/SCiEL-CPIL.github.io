"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO using a Delaunay-based planar graph
    augmented with k-nearest neighbors to ensure connectivity and feasibility.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n, n)
    coordinates : np.ndarray
        Coordinates of shape (n, 2)
    demands : np.ndarray
        Demands of shape (n,)
    capacity : float
        Vehicle capacity

    Returns:
    --------
    np.ndarray
        Heuristic matrix of same shape as distance_matrix
    """
    n = len(demands)
    heuristic = np.full((n, n), 1e-9, dtype=float)
    np.fill_diagonal(heuristic, 0.0)

    if n <= 1:
        return heuristic

    # Get customer indices (excluding depot at index 0)
    customer_indices = np.arange(1, n)
    customer_coords = coordinates[customer_indices]

    try:
        # Build adjacency set for geometrically proximal edges
        adjacency = np.zeros((n, n), dtype=bool)

        # 1) Delaunay triangulation on customers
        tri = Delaunay(customer_coords)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    ci = simplex[i] + 1
                    cj = simplex[j] + 1
                    adjacency[ci, cj] = True
                    adjacency[cj, ci] = True

        # 2) Add k-nearest neighbors for customers to improve local connectivity
        #    and to handle degenerate Delaunay cases
        k = min(5, n - 1)  # keep sparse
        if n > 1:
            tree = cKDTree(coordinates)
            # For each node, get k nearest neighbors (excluding self)
            _, idx = tree.query(coordinates, k=k + 1)
            for i in range(n):
                for j in idx[i]:
                    if i != j:
                        adjacency[i, j] = True

        # 3) Ensure depot has connections to nearest customers
        depot_distances = distance_matrix[0, 1:n]
        k_depot = min(10, n - 1)
        nearest_depot = np.argpartition(depot_distances, k_depot)[:k_depot] + 1
        for idx in nearest_depot:
            adjacency[0, idx] = True
            adjacency[idx, 0] = True

        # Build heuristic: high weight for edges in the sparse graph,
        # using a smooth inverse-distance decay.
        # We use 1 / (1 + dist) to keep values bounded and positive.
        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j]:
                    d = distance_matrix[i, j]
                    if d > 0:
                        heuristic[i, j] = 1.0 / (1.0 + d)
                    else:
                        heuristic[i, j] = 1.0

        # Ensure all off-diagonal entries are at least 1e-9
        off_diag = ~np.eye(n, dtype=bool)
        heuristic[off_diag] = np.maximum(heuristic[off_diag], 1e-9)

    except Exception:
        # Fallback: full inverse-distance heuristic (still valid)
        heuristic = 1.0 / (1.0 + distance_matrix)
        np.fill_diagonal(heuristic, 0.0)
        off_diag = ~np.eye(n, dtype=bool)
        heuristic[off_diag] = np.maximum(heuristic[off_diag], 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
