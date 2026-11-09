import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def get_polygon_from_angles(angles):
        base_angles = 2 * np.pi * np.arange(n) / n
        total_angles = base_angles + angles
        x = np.cos(total_angles)
        y = np.sin(total_angles)
        poly = np.column_stack((x, y))
        
        centroid = np.mean(poly, axis=0)
        poly = poly - centroid
        dists = np.linalg.norm(poly, axis=1)
        mean_d = np.mean(dists)
        if mean_d < 0.01:
            return None, 1.0 # Return penalty
        poly = poly / mean_d
        
        edges = np.roll(poly, -1, axis=0) - poly
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        # Penalty for non-convexity
        # If any cross <= 0, it's not strictly convex.
        # We want to minimize -score + penalty.
        # Penalty = max(0, -min(cross))^2
        min_cross = np.min(cross)
        penalty = 0.0
        if min_cross < 0:
            penalty = (-min_cross) ** 2 * 100.0
            
        return poly, penalty

    def compute_vertex_scores(poly):
        scores = np.zeros(n)
        for i in range(n):
            v = poly[i]
            dists = np.linalg.norm(poly - v, axis=1)
            dists[i] = np.inf
            sorted_dists = np.sort(dists)[:19]
            
            if sorted_dists[0] < 0.05:
                scores[i] = 0.0
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
                scores[i] = 0.0
                continue
                
            ratios = np.maximum(four / D, D / four)
            raw = np.mean(ratios)
            scores[i] = min(1.0, 1.0 / raw)
            
        return scores

    def objective(angles):
        poly, penalty = get_polygon_from_angles(angles)
        if poly is None:
            return 10.0 + penalty
        scores = compute_vertex_scores(poly)
        # Smooth min using log-sum-exp
        # beta controls smoothness. Higher beta -> closer to true min.
        beta = 50.0
        # Avoid log(0) if scores are 0
        scores_safe = np.maximum(scores, 1e-6)
        log_sum_exp = np.log(np.sum(np.exp(-beta * scores_safe)))
        smooth_min = -log_sum_exp / beta
        
        # We want to maximize smooth_min, so minimize -smooth_min
        return -smooth_min + penalty

    # Initial guess: regular polygon (angles = 0)
    best_poly = None
    best_score = -np.inf
    
    # Try multiple starts
    np.random.seed(42)
    starts = [np.zeros(n)]
    # Add some random perturbations
    for _ in range(10):
        starts.append(np.random.uniform(-0.1, 0.1, n))
    
    for start in starts:
        res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.success or res.fun < best_score:
            poly, penalty = get_polygon_from_angles(res.x)
            if poly is not None and penalty < 1e-3: # Check if valid
                scores = compute_vertex_scores(poly)
                min_score = np.min(scores)
                if min_score > best_score:
                    best_score = min_score
                    best_poly = poly

    if best_poly is None:
        # Fallback to regular
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_poly = np.column_stack((np.cos(phi), np.sin(phi)))
        
    return best_poly
