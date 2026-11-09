# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Optimizes a superellipse (Lamé curve) shape to maximize the minimum vertex score.
    """
    start_time = time.time()
    deadline = 600.0 # 600s wall clock limit, leave margin
    
    # Scoring functions
    def normalize(pts):
        centroid = np.mean(pts, axis=0)
        pts_norm = pts - centroid
        dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return None
        return pts_norm / mean_dist

    def vertex_score(v, pts):
        diffs = pts - v
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists > 1e-10
        d = dists[mask]
        if len(d) != 19:
            return 0.0
        
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
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
            
        d_sel = d_sorted[idxs]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)

    def total_score(pts):
        pts_norm = normalize(pts)
        if pts_norm is None:
            return 0.0
        scores = []
        for i in range(len(pts_norm)):
            s = vertex_score(pts_norm[i], pts_norm)
            scores.append(s)
        return np.min(scores)

    def generate_superellipse(params):
        a, b, n = params
        # Ensure valid n
        if n < 1.1: n = 1.1
        angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
        x = np.zeros(20)
        y = np.zeros(20)
        for i, theta in enumerate(angles):
            cx = np.cos(theta)
            sx = np.sin(theta)
            # Handle sign and power
            # If n is very large, 2/n is small.
            x[i] = a * np.sign(cx) * (abs(cx) ** (2.0 / n))
            y[i] = b * np.sign(sx) * (abs(sx) ** (2.0 / n))
        return np.column_stack((x, y))

    def objective(params):
        pts = generate_superellipse(params)
        score = total_score(pts)
        return -score # Minimize negative score

    # Initial guess: Regular polygon (Circle)
    # a=1, b=1, n=2
    x0 = [1.0, 1.0, 2.0]
    
    best_score = -objective(x0)
    best_params = x0
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return generate_superellipse(best_params)

    # Optimization
    # Bounds: a, b in [0.5, 2.0], n in [1.5, 10.0]
    bounds = [(0.5, 2.0), (0.5, 2.0), (1.5, 10.0)]
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x
            
    except Exception:
        pass

    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
            
        # Random params
        a = np.random.uniform(0.8, 1.5)
        b = np.random.uniform(0.8, 1.5)
        n = np.random.uniform(2.0, 6.0)
        x_rand = [a, b, n]
        
        try:
            result = minimize(objective, x_rand, method='Nelder-Mead', bounds=bounds, 
                              options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_params = result.x
                
        except Exception:
            pass

    return generate_superellipse(best_params)

# EVOLVE-BLOCK-END
