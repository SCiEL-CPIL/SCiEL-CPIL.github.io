import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = x[j] - x[i]
                dy1 = y[j] - y[i]
                dx2 = x[k] - x[i]
                dy2 = y[k] - y[i]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    scale = 1.0 / np.sqrt(hull_area)
    points_norm = points * scale
    min_tri = _min_triangle_area(points_norm)
    return min_tri

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    def objective(params):
        current_time = time.time()
        if current_time - start_time > time_limit:
            return 1e9
        points = params.reshape(N, 2)
        if not np.all(np.isfinite(points)):
            return 1e9
        return -_score(points)

    best_score = 0.0
    best_points = None
    
    # Initial guesses
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p14 = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p14.flatten())
    
    # 2. 13 on hull, 1 center
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    p13 = np.column_stack((np.cos(angles13), np.sin(angles13)))
    center = np.array([[0.0, 0.0]])
    p13c = np.vstack((p13, center))
    inits.append(p13c.flatten())
    
    # 3. 12 on hull, 2 near center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    c1 = np.array([[0.0, 0.1]])
    c2 = np.array([[0.0, -0.1]])
    p12cc = np.vstack((p12, c1, c2))
    inits.append(p12cc.flatten())

    # Add noise to break symmetry for each init
    rng = np.random.default_rng(42)
    
    for x0 in inits:
        x0_noisy = x0 + rng.normal(0, 0.01, size=x0.shape)
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
        try:
            result = minimize(objective, x0_noisy, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12})
            
            if result.success or result.fun < -best_score:
                # result.fun is -score
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.reshape(N, 2)
        except Exception:
            continue

    if best_points is None:
        # Fallback
        best_points = inits[0].reshape(N, 2)
        
    # Normalize
    hull_area = _hull_area(best_points)
    if hull_area > 1e-9:
        best_points = best_points / np.sqrt(hull_area)
        
    return best_points
