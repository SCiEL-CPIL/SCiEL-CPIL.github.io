# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Parameterize points by Cartesian coordinates.
    2. Use logarithmic barrier for convexity: enforce positive cross products for sorted angular order.
    3. Optimize to maximize minimum triangle area / hull area using L-BFGS-B with multi-start.
    """
    
    def compute_hull_area(pts):
        """Compute convex hull area using shoelace formula on sorted points."""
        # Sort by angle around centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts[sorted_idx]
        
        # Shoelace formula
        n = len(pts_sorted)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts_sorted[i, 0] * pts_sorted[j, 1]
            area -= pts_sorted[j, 0] * pts_sorted[i, 1]
        return 0.5 * abs(area)
    
    def min_triangle_area(pts):
        """Compute minimum triangle area among all triples."""
        n = len(pts)
        min_area = float('inf')
        for i, j, k in combinations(range(n), 3):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area = area
        return min_area

    def get_sorted_indices(pts):
        """Get indices that sort points by angle around centroid."""
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        return np.argsort(angles)

    def barrier_term(pts, mu):
        """
        Calculate logarithmic barrier term for convexity.
        Assumes points are sorted by angle.
        Returns a large positive value if convexity is violated (cross <= 0).
        """
        idx = get_sorted_indices(pts)
        pts_sorted = pts[idx]
        n = len(pts_sorted)
        barrier = 0.0
        for i in range(n):
            p1 = pts_sorted[i]
            p2 = pts_sorted[(i + 1) % n]
            p3 = pts_sorted[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-8:
                return 1e10 # Penalty for non-convex
            barrier += -np.log(cross)
        return mu * barrier

    def objective(x, mu):
        pts = x.reshape(N, 2)
        
        # Barrier for convexity
        b = barrier_term(pts, mu)
        if b > 1e9:
            return b
            
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
            
        mt_area = min_triangle_area(pts)
        
        # We want to maximize mt_area / hull_area
        # So minimize -(mt_area / hull_area) + barrier
        val = -(mt_area / hull_area)
        return val + b

    # Initial guess: Regular polygon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    x0 = points_init.flatten()
    
    best_points = points_init
    best_score = -min_triangle_area(points_init) / compute_hull_area(points_init)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    for i in range(5): # 5 restarts
        if i == 0:
            x_start = x0.copy()
        else:
            # Perturb initial guess
            noise = rng.normal(0, 0.1, size=N*2)
            x_start = x0 + noise
        
        # Use L-BFGS-B
        # mu controls the barrier strength. Start small, increase? Or just fix?
        # For simplicity, use a fixed moderate mu.
        mu = 1e-3
        
        result = minimize(
            lambda x: objective(x, mu),
            x_start,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        pts_opt = result.x.reshape(N, 2)
        
        # Verify convexity and calculate score
        # Check if points are in convex position
        idx = get_sorted_indices(pts_opt)
        pts_sorted = pts_opt[idx]
        is_convex = True
        for j in range(N):
            p1 = pts_sorted[j]
            p2 = pts_sorted[(j + 1) % N]
            p3 = pts_sorted[(j + 2) % N]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-8:
                is_convex = False
                break
        
        if is_convex:
            hull_area = compute_hull_area(pts_opt)
            if hull_area > 1e-10:
                mt_area = min_triangle_area(pts_opt)
                score = mt_area / hull_area
                if score > best_score:
                    best_score = score
                    best_points = pts_opt

    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = compute_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = best_points.mean(axis=0)
        best_points = (best_points - centroid) * scale
    
    return best_points
# EVOLVE-BLOCK-END
