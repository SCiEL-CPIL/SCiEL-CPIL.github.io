import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # We will use Fourier modes 1, 2, 3, 4 for angular deviations
    # coeffs: [a1, b1, a2, b2, a3, b3, a4, b4]
    n_modes = 4
    dim = 2 * n_modes

    def generate_polygon(coeffs):
        i = np.arange(n)
        base_angles = 2.0 * np.pi * i / n
        
        angles = base_angles.copy()
        for m in range(1, n_modes + 1):
            a = coeffs[2*(m-1)]
            b = coeffs[2*(m-1)+1]
            angles += a * np.cos(2.0 * np.pi * m * i / n) + b * np.sin(2.0 * np.pi * m * i / n)
            
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.column_stack((x, y))
        
        # Normalize
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        dists = np.linalg.norm(pts, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return None
        pts /= mean_dist
        
        return pts

    def score_polygon(pts):
        if pts is None:
            return -1e9
            
        # Check convexity
        edges = np.roll(pts, -1, axis=0) - pts
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        
        # All cross products must have the same sign and be non-zero
        if np.any(np.abs(cross) < 1e-9):
            return -1e9
        if np.any(cross > 0) and np.any(cross < 0):
            return -1e9
            
        # Calculate vertex scores
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        
        min_score = 1.0
        for i in range(n):
            d_i = np.delete(dists[i], i)
            dists_sorted = np.sort(d_i)
            
            if dists_sorted[0] < 0.05:
                min_score = 0.0
                break
            
            gaps = np.diff(dists_sorted)
            min_gap = np.min(gaps)
            k = np.argmin(gaps)
            
            if k == 0:
                d_subset = dists_sorted[0:4]
            elif k == 17: # 19-2 = 17
                d_subset = dists_sorted[15:19]
            else:
                d_subset = dists_sorted[k-1:k+3]
                
            D = (dists_sorted[k] + dists_sorted[k+1]) / 2.0
            if D < 0.1:
                min_score = 0.0
                break
                
            ratios = np.maximum(d_subset / D, D / d_subset)
            raw = np.mean(ratios)
            v_score = min(1.0, 1.0 / raw)
            
            if v_score < min_score:
                min_score = v_score
                
        return min_score

    def objective(coeffs):
        pts = generate_polygon(coeffs)
        s = score_polygon(pts)
        return -s  # Minimize negative score

    # Bounds for coefficients to ensure convexity and avoid singularities
    # Small deviations from regular polygon
    bounds = [(-0.05, 0.05)] * dim
    
    # Use Differential Evolution for global optimization
    # This is more robust than Nelder-Mead for this kind of problem
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            tol=1e-8, 
            polish=True, 
            init='sobol'
        )
        best_coeffs = res.x
        best_score = -res.fun
    except Exception:
        best_coeffs = np.zeros(dim)
        best_score = 0.0

    # Verify and refine with a local search if needed
    # Differential Evolution with polish should be good, but let's add a small local polish
    if best_score > 0:
        try:
            res_local = minimize(
                objective, 
                best_coeffs, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if -res_local.fun > best_score:
                best_score = -res_local.fun
                best_coeffs = res_local.x
        except Exception:
            pass

    final_pts = generate_polygon(best_coeffs)
    final_score = score_polygon(final_pts)
    
    if final_pts is None or final_score <= 0:
        # Fallback to regular polygon
        final_pts = generate_polygon(np.zeros(dim))
        
    return final_pts
