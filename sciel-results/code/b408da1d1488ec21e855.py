"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Use Nelder-Mead optimization on 2D coordinates.
    2. Objective: Maximize (Min Triangle Area / Hull Area).
    3. Constraint: Points must be in convex position. Enforced via a logarithmic 
       barrier on the cross products of the sorted vertices.
    4. Multiple restarts from perturbed regular polygons to escape local minima.
    """
    
    def compute_hull_area_sorted(pts_sorted):
        """Compute convex hull area using shoelace formula on already sorted points."""
        n = len(pts_sorted)
        if n < 3:
            return 0.0
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

    def get_sorted_pts(pts):
        """Sort points by angle around centroid."""
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        return pts[sorted_idx]

    def objective(x, mu=0.1):
        pts = x.reshape(N, 2)
        
        # Sort points by angle around centroid to define the polygon order
        pts_sorted = get_sorted_pts(pts)
        
        # Calculate cross products for the sorted polygon to check convexity
        n = N
        cross_prods = np.zeros(n)
        for i in range(n):
            p1 = pts_sorted[i]
            p2 = pts_sorted[(i + 1) % n]
            p3 = pts_sorted[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            cross_prods[i] = cross
        
        # If any cross product is non-positive, the points are not in strict convex position
        # in this sorted order (or some are collinear/concave).
        if np.any(cross_prods <= 1e-9):
            # Return a large penalty to push away from non-convex regions
            # We use a log barrier term if we were inside, but here we are outside.
            # A smooth penalty is better: sum of squares of negative parts
            neg_cross = cross_prods[cross_prods < 1e-9]
            penalty = np.sum(neg_cross**2) * 1000.0
            # Also compute the ratio to keep the objective continuous-ish if possible, 
            # but for non-convex, the hull area might be ill-defined for the "polygon" 
            # if we just shoelace the sorted points (it will self-intersect or be concave).
            # For non-convex, we just return a large value.
            return 1000.0 + penalty

        # Calculate Hull Area
        hull_area = compute_hull_area_sorted(pts_sorted)
        if hull_area < 1e-9:
            return 1000.0

        # Calculate Min Triangle Area
        mt_area = min_triangle_area(pts)
        
        # Ratio to maximize
        ratio = mt_area / hull_area
        
        # Log barrier to keep cross products positive (smooth repulsion)
        # Only valid if all cross > 0
        log_barrier = -np.sum(np.log(cross_prods))
        
        # Objective: Minimize -ratio + mu * barrier
        return -ratio + mu * log_barrier

    # Initial guess: Regular 13-gon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    
    best_points = points_init
    best_score = -objective(points_init.flatten()) # Note: objective returns negative ratio + barrier
    
    # Run optimization with several perturbations
    np.random.seed(42)
    for _ in range(5):
        # Add small noise to initial points
        noise = np.random.normal(0, 0.05, size=(N, 2))
        x0 = (points_init + noise).flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            opt_pts = result.x.reshape(N, 2)
            
            # Verify convexity and calculate true score
            opt_pts_sorted = get_sorted_pts(opt_pts)
            cross_prods = []
            for i in range(N):
                p1 = opt_pts_sorted[i]
                p2 = opt_pts_sorted[(i + 1) % N]
                p3 = opt_pts_sorted[(i + 2) % N]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                cross_prods.append(cross)
            
            if all(c > 1e-9 for c in cross_prods):
                hull_area = compute_hull_area_sorted(opt_pts_sorted)
                mt_area = min_triangle_area(opt_pts)
                score = mt_area / hull_area
                
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
        except Exception:
            continue

    # Normalize best points: center at origin, scale to unit hull area
    best_pts_sorted = get_sorted_pts(best_points)
    hull_area = compute_hull_area_sorted(best_pts_sorted)
    if hull_area > 1e-9:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = best_points.mean(axis=0)
        best_points = (best_points - centroid) * scale
    
    return best_points
# EVOLVE-BLOCK-END
