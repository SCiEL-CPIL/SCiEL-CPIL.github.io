import numpy as np
import time
from scipy.optimize import minimize

def _score_single_vertex(d_sorted):
    # d_sorted is array of 19 distances
    if np.any(d_sorted < 0.05):
        return 0.0
        
    gaps = np.diff(d_sorted)
    k = np.argmin(gaps)
    
    D = (d_sorted[k] + d_sorted[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17:
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    ds = d_sorted[idxs]
    # Avoid division by zero if D is 0, but D>=0.1
    ratios = np.maximum(ds / D, D / ds)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _score_polygon(points):
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist
    
    n = len(points)
    if n != 20:
        return 0.0
        
    # Check convexity
    # Cross product of consecutive edges
    cross_vals = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i+1) % n]
        p3 = points[(i+2) % n]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-9:
            return 0.0
        cross_vals.append(cross)
        
    # All crosses must have the same sign
    if not (all(c > 0 for c in cross_vals) or all(c < 0 for c in cross_vals)):
        return 0.0
        
    # Check repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                return 0.0
                
    # Compute scores
    scores = []
    for i in range(n):
        v = points[i]
        d = np.linalg.norm(points - v, axis=1)
        d = np.delete(d, i)
        d = np.sort(d)
        s = _score_single_vertex(d)
        scores.append(s)
        
    return min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500.0 # 500s limit, 100s margin
    
    n = 20
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    base_points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _score_polygon(base_points)
    best_points = base_points.copy()
    
    # Optimization function
    def objective(theta_vec):
        # theta_vec is array of 20 angles
        # Ensure they are in [0, 2pi) and sorted?
        # We optimize the differences or just the angles directly?
        # If we optimize angles directly, we might break sorting.
        # Let's optimize the 19 differences + 1 start angle?
        # Or just 20 angles and sort them?
        # Sorting changes the topology if they wrap around.
        # Better: optimize 19 positive gaps that sum to 2pi.
        # Let x be 19 gaps. x_i > 0. sum(x) = 2pi.
        # This is a simplex constraint.
        # We can use a transformation: x_i = exp(y_i) / sum(exp(y)).
        # Then angles are cumulative sums.
        
        if time.time() > deadline:
            return -best_score # Return current best to stop? No, just return something.
            
        y = theta_vec
        # Softplus or exp to ensure positivity
        x = np.exp(y)
        x = x / np.sum(x) * 2 * np.pi
        
        angles = np.cumsum(x)
        # Shift to start at 0?
        # The polygon is rotation invariant.
        # We can fix the first angle to 0 to reduce symmetry breaking?
        # Actually, the parameterization x defines the shape.
        # The absolute rotation doesn't matter.
        # So we just need the relative angles.
        # The cumulative sum starts at x_0.
        # Let's subtract the first angle to fix rotation.
        angles = angles - angles[0]
        
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        s = _score_polygon(points)
        return -s # Minimize negative score
    
    # Initial y for regular polygon
    # Regular gaps are 2pi/20
    # x_reg = 2pi/20
    # y_reg = log(x_reg) ... but x is normalized.
    # If all x are equal, y can be any constant.
    y0 = np.zeros(19)
    
    # Run optimization
    # Use Powell or Nelder-Mead
    # Powell is often robust for this.
    
    for restart in range(5): # 5 restarts
        if time.time() > deadline:
            break
            
        if restart == 0:
            y_start = y0.copy()
        else:
            # Random perturbation
            y_start = y0 + np.random.normal(0, 0.1, 19)
            
        try:
            res = minimize(objective, y_start, method='Powell', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or (hasattr(res, 'fun') and -res.fun > best_score):
                # Reconstruct points
                x = np.exp(res.x)
                x = x / np.sum(x) * 2 * np.pi
                angles = np.cumsum(x)
                angles = angles - angles[0]
                points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
                s = _score_polygon(points)
                if s > best_score:
                    best_score = s
                    best_points = points
        except Exception:
            pass
            
    return best_points
