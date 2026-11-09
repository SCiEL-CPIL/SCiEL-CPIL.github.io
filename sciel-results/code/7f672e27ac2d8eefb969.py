# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # safety margin before 630s limit

    def min_triangle_area(pts):
        """Compute the minimum triangle area among all C(n,3) triangles."""
        n_pts = len(pts)
        min_area = float('inf')
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                    if area < min_area:
                        min_area = area
        return min_area

    def hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return float('inf')

    def score(pts):
        """Compute the ratio A_min / A_hull."""
        try:
            a_min = min_triangle_area(pts)
            a_hull = hull_area(pts)
            if a_hull <= 0 or not np.isfinite(a_hull):
                return 0.0
            return a_min / a_hull
        except Exception:
            return 0.0

    def neg_score(pts_flat):
        """Negative score for minimization."""
        pts = pts_flat.reshape(-1, 2)
        return -score(pts)

    # Strategy: Lloyd's algorithm + optimization
    # Start with a regular polygon for hull points, interior points via Lloyd relaxation
    
    best_points = None
    best_score_val = 0.0

    # Initialize with multiple random starting configurations
    rng = np.random.default_rng(seed=42)
    
    # Configuration 1: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Configuration 2: Grid-like arrangement
    init2 = rng.random((n, 2))
    init2 = (init2 - init2.mean(axis=0)) / init2.std(axis=0)
    
    # Configuration 3: Concentric pattern
    r1 = 0.3
    r2 = 0.8
    angles_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    angles_outer = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    inner = np.column_stack([r1 * np.cos(angles_inner), r1 * np.sin(angles_inner)])
    outer = np.column_stack([r2 * np.cos(angles_outer), r2 * np.sin(angles_outer)])
    init3 = np.vstack([inner, outer])
    
    # Configuration 4: Another random
    init4 = rng.random((n, 2))
    init4 = (init4 - init4.mean(axis=0)) / init4.std(axis=0)

    inits = [init1, init2, init3, init4]

    for init in inits:
        if time.time() > deadline - 30:
            break

        # Lloyd's algorithm: relax interior points toward Voronoi centroids
        # First, identify which points are on the hull
        try:
            hull = ConvexHull(init)
            hull_indices = set(hull.vertices)
            interior_indices = [i for i in range(n) if i not in hull_indices]
        except Exception:
            interior_indices = list(range(n))

        # Lloyd relaxation for interior points
        points = init.copy()
        for iteration in range(200):
            if time.time() > deadline - 30:
                break
            
            try:
                # Compute Delaunay triangulation
                tri = Delaunay(points)
                
                # For each interior point, compute its Voronoi cell centroid
                # Approximate by averaging neighbors in Delaunay
                for idx in interior_indices:
                    # Find neighbors of this point in Delaunay
                    neighbors = []
                    for simplex in tri.simplices:
                        if idx in simplex:
                            for other in simplex:
                                if other != idx:
                                    neighbors.append(other)
                    
                    if neighbors:
                        neighbor_pts = points[neighbors]
                        centroid = neighbor_pts.mean(axis=0)
                        # Move point toward centroid
                        points[idx] = points[idx] + 0.1 * (centroid - points[idx])
            except Exception:
                break

        # Now optimize all points using scipy
        x0 = points.flatten()
        
        # Try BFGS optimization
        try:
            result = minimize(neg_score, x0, method='BFGS', 
                              options={'maxiter': 500, 'gtol': 1e-6})
            if result.success or result.fun < -best_score_val:
                candidate = result.x.reshape(-1, 2)
                s = score(candidate)
                if s > best_score_val:
                    best_score_val = s
                    best_points = candidate
        except Exception:
            pass

        # Also try L-BFGS-B
        if time.time() < deadline - 30:
            try:
                result = minimize(neg_score, x0, method='L-BFGS-B',
                                  options={'maxiter': 1000, 'gtol': 1e-6})
                if result.fun < -best_score_val:
                    candidate = result.x.reshape(-1, 2)
                    s = score(candidate)
                    if s > best_score_val:
                        best_score_val = s
                        best_points = candidate
            except Exception:
                pass

    # If no valid result, return a regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
