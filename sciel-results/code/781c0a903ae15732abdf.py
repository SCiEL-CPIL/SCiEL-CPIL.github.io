import numpy as np
import scipy.optimize as opt
import time

def compute_score(points):
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist
    
    n = len(points)
    min_score = 1.0
    
    for i in range(n):
        v = points[i]
        # Distances to other vertices
        dists = np.linalg.norm(points - v, axis=1)
        dists[i] = np.inf # Exclude self
        sorted_dists = np.sort(dists)
        
        # Find largest gap? No, "first index minimizing d_{k+1} - d_k"
        # This means smallest gap.
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        
        # Check D
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2
        if D < 0.1:
            return 0.0 # Vertex scores 0, so min score is 0
        
        # Select 4 distances
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == n - 2: # 18
            idx = [n-5, n-4, n-3, n-2] # 15..18
        else:
            idx = [k-1, k, k+1, k+2]
            
        d_vals = sorted_dists[idx]
        
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_vals / D, D / d_vals)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
        
    return min_score

def objective(radii):
    # radii: array of 20 positive values
    if np.any(radii <= 0):
        return 1e6
    
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Convexity check
    # Cross product of consecutive edges
    cross_min = np.inf
    for i in range(20):
        p0 = points[i]
        p1 = points[(i+1)%20]
        p2 = points[(i+2)%20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross < cross_min:
            cross_min = cross
            
    penalty = 0.0
    if cross_min < 0:
        penalty = -cross_min * 100.0 # Heavy penalty for non-convexity
        
    score = compute_score(points)
    return -score + penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    timeout = 580.0 # Safety margin
    
    # Initial guess: Alternating 1.0, 0.8
    radii_init = np.ones(20)
    for i in range(1, 20, 2):
        radii_init[i] = 0.8
        
    # Optimization
    result = None
    try:
        # Use Nelder-Mead as it doesn't need gradients and handles non-smoothness better
        res = opt.minimize(objective, radii_init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < 0.672: # 0.672 is approx score of regular 20-gon
            result = res.x
    except Exception:
        pass
        
    # If result is None or bad, try L-BFGS-B
    if result is None:
        try:
            res = opt.minimize(objective, radii_init, method='L-BFGS-B', 
                               options={'maxiter': 5000, 'ftol': 1e-8})
            if res.fun < 0.672:
                result = res.x
        except Exception:
            pass
            
    if result is None:
        # Fallback to regular polygon
        result = np.ones(20)
        
    # Ensure convexity and validity
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x = result * np.cos(angles)
    y = result * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Final check
    if compute_score(points) == 0.0:
         # If invalid, return regular
         points = np.column_stack((np.cos(angles), np.sin(angles)))
         
    return points

