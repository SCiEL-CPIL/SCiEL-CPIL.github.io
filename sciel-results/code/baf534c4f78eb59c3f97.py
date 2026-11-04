"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import ConvexHull


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Use a smooth approximation (Log-Sum-Exp) for the minimum triangle area.
    2. Enforce convex position via a barrier term on cross products of angularly sorted points.
    3. Optimize using Powell's method which is robust for smooth, non-convex landscapes.
    4. Vectorize area calculations for speed.
    """
    
    # Generate regular 13-gon centered at origin as initial guess
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    
    # Pre-compute indices for all triples
    triplets = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triplets])
    idx_j = np.array([t[1] for t in triplets])
    idx_k = np.array([t[2] for t in triplets])
    
    def compute_hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10

    def get_areas(pts):
        """
        Compute areas for all pre-defined triples.
        Returns array of shape (num_triplets,)
        """
        p_i = pts[idx_i]
        p_j = pts[idx_j]
        p_k = pts[idx_k]
        
        cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
        areas = 0.5 * np.abs(cross)
        return areas

    def objective(x):
        pts = x.reshape(N, 2)
        
        # 1. Convexity Check and Barrier
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts[sorted_idx]
        
        barrier = 0.0
        for i in range(N):
            p1 = pts_sorted[i]
            p2 = pts_sorted[(i + 1) % N]
            p3 = pts_sorted[(i + 2) % N]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross < 1e-6:
                barrier += (1e-6 - cross) ** 2 * 1e6
            else:
                barrier -= np.log(cross + 1e-10) * 0.1
                
        if barrier > 1e5:
            return 1e6 + barrier

        # 2. Compute Areas
        areas = get_areas(pts)
        
        # 3. Compute Hull Area
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
        
        # 4. Smooth Minimum Area using Log-Sum-Exp
        beta = 100.0
        normalized_areas = areas / hull_area
        
        m = np.max(normalized_areas)
        exp_terms = np.exp(-beta * (normalized_areas - m))
        sum_exp = np.sum(exp_terms)
        obj_val = -m + (1.0 / beta) * np.log(sum_exp)
        
        total_obj = obj_val + barrier
        
        return total_obj

    # Start from regular polygon
    x0 = points_init.flatten()
    
    # Use Powell optimization
    result = minimize(
        objective,
        x0,
        method='Powell',
        options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
    )
    
    optimized_points = result.x.reshape(N, 2)
    
    # Fallback if optimization failed badly
    if result.fun > 1e5:
        optimized_points = points_init
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = compute_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
    
    return optimized_points
# EVOLVE-BLOCK-END
