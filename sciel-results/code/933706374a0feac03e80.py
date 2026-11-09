import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # Fourier series parameters
    # We restrict to modes 1, 2, 3 to enforce smoothness and reduce dimensionality.
    # theta_i = 2*pi*i/n + sum( a_m * cos(2*pi*m*i/n) + b_m * sin(2*pi*m*i/n) ) for m in 1, 2, 3
    # Total 6 parameters.
    
    def generate_polygon(coeffs):
        # coeffs: [a1, b1, a2, b2, a3, b3]
        i = np.arange(n)
        base_angles = 2.0 * np.pi * i / n
        
        angles = base_angles.copy()
        for m in range(1, 4):
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
        # Calculate cross products of consecutive edges
        edges = np.roll(pts, -1, axis=0) - pts
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        
        # All cross products must have the same sign and be non-zero
        if np.any(np.abs(cross) < 1e-9):
            return -1e9
        if np.any(cross > 0) and np.any(cross < 0):
            return -1e9
            
        # Calculate vertex scores
        min_score = 1.0
        for i in range(n):
            v = pts[i]
            # Distances to all other vertices
            diffs = pts - v
            dists = np.linalg.norm(diffs, axis=1)
            # Exclude self
            dists = np.delete(dists, i)
            
            # Sort
            dists_sorted = np.sort(dists)
            
            # Check if any dist < 0.05
            if dists_sorted[0] < 0.05:
                min_score = 0.0
                break # No need to check others if min is 0
            
            # Find k: first index minimizing d[k+1] - d[k]
            gaps = np.diff(dists_sorted)
            min_gap = np.min(gaps)
            k = np.argmin(gaps)
            
            # Determine which 4 distances to use
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
                
            # raw = mean of max(d/D, D/d)
            ratios = np.maximum(d_subset / D, D / d_subset)
            raw = np.mean(ratios)
            v_score = min(1.0, 1.0 / raw)
            
            if v_score < min_score:
                min_score = v_score
                
        return min_score

    def objective(coeffs):
        pts = generate_polygon(coeffs)
        s = score_polygon(pts)
        return -s # Minimize negative score

    # We will run multiple random starts to find a good solution.
    # The regular polygon (coeffs=0) is a valid starting point.
    # We'll try a few random perturbations.
    
    best_score = -1e9
    best_coeffs = np.zeros(6)
    
    # Regular polygon
    reg_pts = generate_polygon(best_coeffs)
    reg_score = score_polygon(reg_pts)
    if reg_score > best_score:
        best_score = reg_score
        best_coeffs = np.zeros(6)
        
    # Random starts
    np.random.seed(42)
    for seed in range(50):
        np.random.seed(seed)
        # Small perturbations to break symmetry
        x0 = np.random.normal(0, 0.05, size=6)
        
        # Use Nelder-Mead for robustness in small dimensions
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < -best_score: # -fun is the score
            cur_score = -res.fun
            if cur_score > best_score:
                best_score = cur_score
                best_coeffs = res.x
                
    # Final check and return
    final_pts = generate_polygon(best_coeffs)
    final_score = score_polygon(final_pts)
    
    # If optimization failed or went invalid, return regular polygon
    if final_pts is None or final_score <= 0:
        final_pts = generate_polygon(np.zeros(6))
        
    return final_pts
