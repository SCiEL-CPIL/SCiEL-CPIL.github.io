import numpy as np
from scipy.optimize import differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def get_polygon_from_angles(angles):
        # angles: perturbations from regular polygon
        # base angles: 2*pi*i/n
        base_angles = 2 * np.pi * np.arange(n) / n
        total_angles = base_angles + angles
        
        # Sort angles to ensure cyclic order? 
        # If perturbations are small, order is preserved.
        # If not, we might get self-intersecting or non-convex.
        # We assume small perturbations.
        
        x = np.cos(total_angles)
        y = np.sin(total_angles)
        poly = np.column_stack((x, y))
        
        # Normalize
        centroid = np.mean(poly, axis=0)
        poly = poly - centroid
        dists = np.linalg.norm(poly, axis=1)
        mean_d = np.mean(dists)
        if mean_d < 0.01:
            return None
        poly = poly / mean_d
        
        # Check strict convexity
        edges = np.roll(poly, -1, axis=0) - poly
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        # For CCW, all cross should be > 0
        if np.any(cross <= 0):
            return None
            
        return poly

    def compute_scores(poly):
        scores = []
        for i in range(n):
            v = poly[i]
            dists = np.linalg.norm(poly - v, axis=1)
            dists[i] = np.inf
            sorted_dists = np.sort(dists)[:19]
            
            if sorted_dists[0] < 0.05:
                scores.append(0.0)
                continue
                
            gaps = np.diff(sorted_dists)
            k = np.argmin(gaps)
            
            if k == 0:
                D = (sorted_dists[0] + sorted_dists[1]) / 2.0
                four = sorted_dists[0:4]
            elif k == 17:
                D = (sorted_dists[17] + sorted_dists[18]) / 2.0
                four = sorted_dists[15:19]
            else:
                D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
                four = sorted_dists[k-1:k+3]
                
            if D < 0.1:
                scores.append(0.0)
                continue
                
            ratios = np.maximum(four / D, D / four)
            raw = np.mean(ratios)
            scores.append(min(1.0, 1.0 / raw))
            
        return np.min(scores)

    def objective(angles):
        poly = get_polygon_from_angles(angles)
        if poly is None:
            return 10.0
        score = compute_scores(poly)
        return -score

    # Define bounds for angles
    # Max perturbation should be less than half the step size (pi/10) to maintain order
    # Step is 2*pi/20 = pi/10. Half is pi/20.
    # Let's use a slightly smaller bound to be safe, e.g., pi/25
    bound = np.pi / 25
    bounds = [(-bound, bound) for _ in range(n)]
    
    # Use differential evolution
    # tol=1e-4, maxiter=200
    res = differential_evolution(objective, bounds, seed=42, maxiter=200, tol=1e-4, polish=True)
    
    best_poly = get_polygon_from_angles(res.x)
    
    if best_poly is None:
        # Fallback
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_poly = np.column_stack((np.cos(phi), np.sin(phi)))
        
    return best_poly
