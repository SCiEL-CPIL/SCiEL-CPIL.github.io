# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13

    def min_triangle_area_vectorized(points):
        """Compute the minimum triangle area among all triples of points using vectorization."""
        # points shape: (n, 2)
        # We need to compute areas for all i < j < k
        # Using broadcasting:
        # p1 = points[:, np.newaxis, np.newaxis, :]  # (n, 1, 1, 2)
        # p2 = points[np.newaxis, :, np.newaxis, :]  # (1, n, 1, 2)
        # p3 = points[np.newaxis, np.newaxis, :, :]  # (1, 1, n, 2)
        
        # To save memory and time, we can iterate over i and vectorize j, k
        # Or just use the full triple loop if n is small (13 is small, 286 triangles).
        # Vectorizing fully creates a 13x13x13 array which is fine.
        
        p1 = points[:, np.newaxis, np.newaxis, :]
        p2 = points[np.newaxis, :, np.newaxis, :]
        p3 = points[np.newaxis, np.newaxis, :, :]
        
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # Cross product in 2D: (p2-p1) x (p3-p1)
        # Let v1 = p2 - p1, v2 = p3 - p1
        # Area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
        
        v1 = p2 - p1 # (n, n, 1, 2)
        v2 = p3 - p1 # (n, 1, n, 2)
        
        # We need v1_x * v2_y - v1_y * v2_x
        # v1_x shape: (n, n, 1)
        # v2_y shape: (n, 1, n)
        # Result shape: (n, n, n)
        
        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        areas = np.abs(cross) * 0.5
        
        # We only want i < j < k
        # Create a mask
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        
        valid_areas = areas[mask]
        if len(valid_areas) == 0:
            return 0.0
        return np.min(valid_areas)

    def hull_area(points):
        """Compute the area of the convex hull of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def score(points):
        """Score = A_min / A_hull. Higher is better."""
        a_hull = hull_area(points)
        if a_hull <= 0:
            return 0.0
        a_min = min_triangle_area_vectorized(points)
        return a_min / a_hull

    def negative_score(points_flat):
        """Negative score for minimization."""
        points = points_flat.reshape(n, 2)
        # Check for NaN or inf
        if not np.all(np.isfinite(points)):
            return 1e10
        s = score(points)
        return -s

    def optimize_from_start(start_points):
        """Run Nelder-Mead optimization from a given start."""
        # Normalize start to unit hull area to help convergence
        a_hull = hull_area(start_points)
        if a_hull > 0:
            start_points = start_points / np.sqrt(a_hull)
        
        result = minimize(
            negative_score,
            start_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.success or np.isfinite(result.fun):
            points_opt = result.x.reshape(n, 2)
            if np.all(np.isfinite(points_opt)):
                try:
                    ConvexHull(points_opt)
                    return points_opt, score(points_opt)
                except Exception:
                    pass
        return start_points, 0.0

    # Generate diverse initial configurations
    np.random.seed(42)
    starts = []
    
    # 1. Regular 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(np.vstack([p1, np.array([0.0, 0.0])]))
    
    # 2. Regular 6-gon + Regular 6-gon (rotated) + center
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles6), np.sin(angles6)])
    inner_angles = angles6 + np.pi / 6
    inner = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)]) * 0.5
    starts.append(np.vstack([outer, inner, np.array([0.0, 0.0])]))
    
    # 3. Regular 5-gon + Regular 7-gon
    angles5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p5 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    p7 = np.column_stack([np.cos(angles7), np.sin(angles7)]) * 0.6
    starts.append(np.vstack([p5, p7]))
    
    # 4. Random points in disk
    for _ in range(5):
        r = np.sqrt(np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        p_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        starts.append(p_rand)
        
    # 5. Grid-like
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    # Take 13 points from a 4x4 grid (16 points)
    starts.append(grid_points[:13])
    
    # Run optimization for each start
    best_points = starts[0]
    best_score = -np.inf
    
    for start in starts:
        pts, sc = optimize_from_start(start)
        if sc > best_score:
            best_score = sc
            best_points = pts
            
    # Final verification
    if not np.all(np.isfinite(best_points)):
        # Fallback
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = np.vstack([best_points, np.array([0.0, 0.0])])
        
    try:
        ConvexHull(best_points)
    except Exception:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = np.vstack([best_points, np.array([0.0, 0.0])])
        
    return best_points

# EVOLVE-BLOCK-END
