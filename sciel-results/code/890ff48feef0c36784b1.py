import numpy as np
from scipy.optimize import minimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Angles for regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Precompute triangle indices
    tri_indices = list(itertools.combinations(range(n), 3))
    tri_indices = np.array(tri_indices)
    
    def get_points(radii):
        # radii: (13,)
        # points: (13, 2)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))
    
    def compute_score(radii, beta):
        points = get_points(radii)
        
        # Hull Area (Polygon Area)
        # Since angles are regular and sorted, the hull is the polygon in this order
        # unless radii vary wildly causing self-intersection or concavity.
        # For small perturbations, it's convex.
        x = points[:, 0]
        y = points[:, 1]
        # Shoelace formula
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area_hull = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        
        if area_hull < 1e-9:
            return 0.0 # Avoid division by zero
            
        # Triangle Areas
        # Vectorized calculation
        # For each triangle (i, j, k), area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # Let's extract points for all triangles
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        
        # Cross product of (pj - pi) and (pk - pi)
        # z-component: (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        
        # Soft min
        # -1/beta * log(sum(exp(-beta * areas)))
        # To avoid overflow/underflow, subtract max area? No, we want min.
        # min(a) = -1/beta log(sum(exp(-beta a)))
        # Let m = min(areas). Then min(a) = m - 1/beta log(sum(exp(-beta (a - m))))
        m = np.min(areas)
        # If m is very small, soft min is close to 0
        # exp(-beta * (areas - m)) is <= 1, so sum is <= 286
        # This is stable.
        soft_min_area = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - m))))
        
        return soft_min_area / area_hull

    def objective(radii, beta):
        return -compute_score(radii, beta)
    
    best_score = -np.inf
    best_radii = np.ones(n)
    
    # Restart with different initial radii and betas
    rng = np.random.default_rng(42)
    
    # List of betas to anneal
    betas = [10, 50, 100, 500, 1000, 5000]
    
    for beta in betas:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            if restart == 0:
                r0 = np.ones(n)
            else:
                # Small perturbation around best found or random
                r0 = best_radii + rng.normal(0, 0.05, n)
                r0 = np.clip(r0, 0.5, 2.0) # Keep radii reasonable
            
            res = minimize(
                objective, 
                r0, 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_radii = res.x
        
        # Check time
        if time.time() > deadline - 60:
            break
            
    # Final points
    points = get_points(best_radii)
    
    # Optional: One final polish with very high beta on the full 26 coordinates?
    # The radii-only model might be too restrictive if the optimum is not star-shaped with respect to the center.
    # However, for N=13, the regular polygon is the symmetric optimum.
    # Let's try to polish with full coordinates if time permits.
    
    if time.time() < deadline - 30:
        # Convert to full coordinate optimization
        # Objective: -soft_min_area / hull_area
        # Hull area is non-diff. 
        # We can use the polygon area formula if we assume convex position.
        # If the points become non-convex, the polygon area formula is wrong.
        # We can check convexity.
        
        def compute_score_full(points_flat, beta):
            pts = points_flat.reshape(n, 2)
            
            # Check if convex?
            # For now, assume convex and use polygon area.
            # If this fails, we fall back to the radii solution.
            
            x = pts[:, 0]
            y = pts[:, 1]
            
            # Compute Hull Area using ConvexHull for accuracy in score, 
            # but for gradient, we need diff.
            # Let's use the polygon area for the gradient, and ConvexHull for the true score.
            
            # Polygon area (assumes convex position)
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            area_poly = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
            
            if area_poly < 1e-9:
                return 0.0
            
            # Triangle areas
            pi = pts[tri_indices[:, 0]]
            pj = pts[tri_indices[:, 1]]
            pk = pts[tri_indices[:, 2]]
            
            dx1 = pj[:, 0] - pi[:, 0]
            dy1 = pj[:, 1] - pi[:, 1]
            dx2 = pk[:, 0] - pi[:, 0]
            dy2 = pk[:, 1] - pi[:, 1]
            
            cross = dx1 * dy2 - dy1 * dx2
            areas = 0.5 * np.abs(cross)
            
            m = np.min(areas)
            soft_min_area = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - m))))
            
            return soft_min_area / area_poly
        
        def objective_full(points_flat, beta):
            return -compute_score_full(points_flat, beta)
        
        # Start from best radii points
        p0 = points.flatten()
        
        # Try a few steps with high beta
        for beta in [1000, 5000, 10000]:
            if time.time() > deadline - 10:
                break
            res = minimize(
                objective_full, 
                p0, 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                # Check if this is actually better using true ConvexHull
                pts_new = res.x.reshape(n, 2)
                try:
                    hull = ConvexHull(pts_new)
                    area_hull_true = hull.volume
                    # Compute true min area
                    pi = pts_new[tri_indices[:, 0]]
                    pj = pts_new[tri_indices[:, 1]]
                    pk = pts_new[tri_indices[:, 2]]
                    dx1 = pj[:, 0] - pi[:, 0]
                    dy1 = pj[:, 1] - pi[:, 1]
                    dx2 = pk[:, 0] - pi[:, 0]
                    dy2 = pk[:, 1] - pi[:, 1]
                    cross = dx1 * dy2 - dy1 * dx2
                    areas = 0.5 * np.abs(cross)
                    true_min = np.min(areas)
                    true_score = true_min / area_hull_true
                    
                    if true_score > best_score:
                        best_score = true_score
                        points = pts_new
                        p0 = res.x
                except Exception:
                    pass
        
    return points
