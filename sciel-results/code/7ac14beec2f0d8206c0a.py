"""Heuristic matrix for ant colony optimization of the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, ConvexHull
from scipy.linalg import eigh


def _mds_coordinates(d, n):
    """Compute 2D MDS coordinates from a distance matrix."""
    try:
        # Double centering
        ones = np.ones((n, n)) / n
        J = np.eye(n) - ones
        B = -0.5 * J @ (d ** 2) @ J
        # Eigen decomposition
        eigvals, eigvecs = eigh(B)
        # Take the two largest eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Ensure non-negative eigenvalues
        eigvals = np.maximum(eigvals, 0)
        coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
        return coords
    except Exception:
        return None


def _two_opt_improve(tour, dist, max_passes=50):
    """Improve a tour using 2-opt moves."""
    tour = list(tour)
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == n - 1 and i == 1:
                    continue
                # Calculate the change in tour length
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new + 1e-10 < old:
                    # Reverse the segment
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def _greedy_tour(dist, start=0, n=None):
    """Construct a tour using nearest neighbor heuristic."""
    if n is None:
        n = dist.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        # Find nearest unvisited node
        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j] and dist[current, j] < min_dist:
                min_dist = dist[current, j]
                next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Ensure the matrix is square and valid
    if d.shape[0] != d.shape[1]:
        return np.ones_like(d)
    
    # Work on a copy to avoid modifying the original
    d_safe = d.copy()
    np.fill_diagonal(d_safe, np.inf)
    
    # Compute MDS coordinates for geometric reasoning
    coords = _mds_coordinates(d_safe, n)
    
    # Build heuristic from multiple components
    # 1. Base distance-based heuristic
    # Use a moderate exponent to balance local vs global structure
    alpha = 2.0 
    eta_base = np.full((n, n), 1e-9)
    
    # Vectorized distance power calculation
    # Avoid division by zero
    d_safe_calc = d_safe.copy()
    d_safe_calc[d_safe_calc == 0] = np.inf
    eta_base = 1.0 / (d_safe_calc ** alpha)
    # Set diagonal to 0 (or 1e-9 later)
    np.fill_diagonal(eta_base, 0)
    
    # 2. Delaunay triangulation edges
    if coords is not None:
        try:
            tri = Delaunay(coords)
            delaunay_mask = np.zeros((n, n), dtype=bool)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u, v = simplex[i], simplex[j]
                        delaunay_mask[u, v] = True
                        delaunay_mask[v, u] = True
            
            # Boost Delaunay edges significantly
            eta_base[delaunay_mask] *= 2.0
        except Exception:
            pass
        
        # 3. Convex Hull edges
        try:
            hull = ConvexHull(coords)
            hull_mask = np.zeros((n, n), dtype=bool)
            for simplex in hull.simplices:
                # For 2D, simplices are edges (pairs of points)
                u, v = simplex[0], simplex[1]
                hull_mask[u, v] = True
                hull_mask[v, u] = True
            
            # Boost hull edges even more, as they are critical for the perimeter
            eta_base[hull_mask] *= 2.0
        except Exception:
            pass

    # 4. Multi-start greedy tours with 2-opt
    best_tour = None
    best_len = np.inf
    num_starts = 10
    
    for start in range(min(num_starts, n)):
        if time.time() > deadline - 30:
            break
        tour = _greedy_tour(d_safe, start=start, n=n)
        tour = _two_opt_improve(tour, d_safe, max_passes=50)
        
        # Calculate tour length
        length = 0
        for i in range(n):
            length += d_safe[tour[i], tour[(i + 1) % n]]
        
        if length < best_len:
            best_len = length
            best_tour = tour
    
    # 5. Boost edges in the best tour
    if best_tour is not None:
        tour_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            u, v = best_tour[i], best_tour[(i + 1) % n]
            tour_mask[u, v] = True
            tour_mask[v, u] = True
        # Strong boost for the heuristic tour
        eta_base[tour_mask] *= 3.0

    # Final safety check
    eta_base = np.where(np.isfinite(eta_base) & (eta_base > 0), eta_base, 1e-9)
    np.fill_diagonal(eta_base, 1e-9)
    
    return eta_base
# EVOLVE-BLOCK-END
