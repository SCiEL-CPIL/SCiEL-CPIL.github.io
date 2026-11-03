import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _all_areas(points):
    n = points.shape[0]
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    return np.array(areas)

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _objective(flat):
    """
    Minimize -score.
    flat shape: (3, 2) for the anchor points.
    The other 11 points are fixed at a regular 11-gon scaled to have hull area 1.
    """
    anchors = flat.reshape(3, 2)
    
    # Create the 11-gon
    angles = np.linspace(0, 2*np.pi, 11, endpoint=False)
    # Scale factor r such that area of 11-gon is 1.
    # Area of regular n-gon with circumradius r is (n/2) * r^2 * sin(2*pi/n)
    n_gon = 11
    # r^2 = 2 / (n * sin(2*pi/n))
    r2 = 2.0 / (n_gon * np.sin(2*np.pi/n_gon))
    r = np.sqrt(r2)
    
    gon_points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    
    points = np.vstack([anchors, gon_points])
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    score = _score(points)
    if score < 1e-12:
        return 1e10
        
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf

    # Initial guess for anchors: points inside the 11-gon, roughly forming a triangle
    # Center is (0,0). 11-gon vertices are on circle r~1.
    # Let's try anchors at radius 0.5, angles 0, 2pi/3, 4pi/3
    r_anchor = 0.5
    ang_anchor = np.array([0, 2*np.pi/3, 4*np.pi/3])
    init_anchors = np.column_stack((r_anchor * np.cos(ang_anchor), r_anchor * np.sin(ang_anchor)))
    
    # Run Nelder-Mead to optimize the anchors
    try:
        res = minimize(
            _objective,
            init_anchors.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if res.fun < 1e9:
            anchors = res.x.reshape(3, 2)
            # Reconstruct full set to get score
            angles = np.linspace(0, 2*np.pi, 11, endpoint=False)
            r2 = 2.0 / (11 * np.sin(2*np.pi/11))
            r = np.sqrt(r2)
            gon_points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
            points = np.vstack([anchors, gon_points])
            
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    except Exception:
        pass

    # If the first run didn't yield a good result, try a few random starts
    if best_score < 1.0: # Heilbronn number for n=14 is around 0.02-0.05? 
        # Actually, the score is min_area / hull_area.
        # For regular 14-gon, min area is small.
        # Let's just run a few more random starts if we have time.
        np.random.seed(42)
        for i in range(5):
            if time.time() > deadline:
                break
            
            # Random anchors inside the unit circle
            r_rand = np.random.uniform(0.1, 0.8, 3)
            th_rand = np.random.uniform(0, 2*np.pi, 3)
            init_anchors = np.column_stack((r_rand * np.cos(th_rand), r_rand * np.sin(th_rand)))
            
            try:
                res = minimize(
                    _objective,
                    init_anchors.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res.fun < 1e9:
                    anchors = res.x.reshape(3, 2)
                    angles = np.linspace(0, 2*np.pi, 11, endpoint=False)
                    r2 = 2.0 / (11 * np.sin(2*np.pi/11))
                    r = np.sqrt(r2)
                    gon_points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
                    points = np.vstack([anchors, gon_points])
                    
                    score = _score(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        
            except Exception:
                pass

    # Fallback to regular 14-gon if nothing worked
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)

    return best_points
