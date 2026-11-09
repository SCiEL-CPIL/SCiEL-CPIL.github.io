import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
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
        edges = np.roll(pts, -1, axis=0) - pts
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        
        # All cross products must have the same sign and be non-zero
        if np.any(np.abs(cross) < 1e-9):
            return -1e9
        if np.any(cross > 0) and np.any(cross < 0):
            return -1e9
            
        # Calculate vertex scores vectorized
        # Pairwise distances
        # dists[i, j] is distance between pts[i] and pts[j]
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        
        min_score = 1.0
        for i in range(n):
            # Distances to all other vertices
            d_i = np.delete(dists[i], i)
            dists_sorted = np.sort(d_i)
            
            # Check if any dist < 0.05
            if dists_sorted[0] < 0.05:
                min_score = 0.0
                break
            
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

    best_score = -1e9
    best_coeffs = np.zeros(6)
    
    # Regular polygon
    reg_pts = generate_polygon(best_coeffs)
    reg_score = score_polygon(reg_pts)
    if reg_score > best_score:
        best_score = reg_score
        best_coeffs = np.zeros(6)
        
    # Define a set of initial points to explore different basins
    starts = []
    
    # Random starts with varied scales
    for seed in range(200):
        np.random.seed(seed)
        scale = 0.05 + 0.15 * np.random.rand()
        x0 = np.random.normal(0, scale, size=6)
        starts.append(x0)
        
    # Structured starts: Try to enhance specific Fourier modes
    for m in range(1, 4):
        for scale in [0.1, 0.15, 0.2, 0.25]:
            # Cosine mode
            x0 = np.zeros(6)
            x0[2*(m-1)] = scale
            starts.append(x0)
            # Sine mode
            x0 = np.zeros(6)
            x0[2*(m-1)+1] = scale
            starts.append(x0)
            # Mixed
            x0 = np.zeros(6)
            x0[2*(m-1)] = scale * 0.7
            x0[2*(m-1)+1] = scale * 0.7
            starts.append(x0)
    
    # Run optimization for each start
    for x0 in starts:
        # Use Nelder-Mead with tighter tolerances
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True, 'maxfev': 10000})
            cur_score = -res.fun
            if cur_score > best_score:
                best_score = cur_score
                best_coeffs = res.x
        except Exception:
            pass
    
    # Local refinement: try small perturbations around the best solution
    if best_score > 0:
        np.random.seed(42)
        for _ in range(50):
            perturbation = np.random.normal(0, 0.01, size=6)
            trial_coeffs = best_coeffs + perturbation
            try:
                trial_score = -objective(trial_coeffs)
                if trial_score > best_score:
                    best_score = trial_score
                    best_coeffs = trial_coeffs
            except Exception:
                pass
    
    # Final check and return
    final_pts = generate_polygon(best_coeffs)
    final_score = score_polygon(final_pts)
    
    # If optimization failed or went invalid, return regular polygon
    if final_pts is None or final_score <= 0:
        final_pts = generate_polygon(np.zeros(6))
        
    return final_pts
