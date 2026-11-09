import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r0 = np.ones(n)
    
    def get_points(r):
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)
        
    def compute_scores(points):
        scores = np.zeros(n)
        for i in range(n):
            diffs = points - points[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists = np.delete(dists, i)
            dists = np.sort(dists)
            
            if dists[0] < 0.05:
                scores[i] = 0.0
                continue
                
            gaps = np.diff(dists)
            k = np.argmin(gaps)
            
            D = (dists[k] + dists[k+1]) / 2.0
            if D < 0.1:
                scores[i] = 0.0
                continue
                
            if k == 0:
                d_window = dists[0:4]
            elif k == 17:
                d_window = dists[15:19]
            else:
                d_window = dists[k-1:k+3]
                
            ratios = np.maximum(d_window / D, D / d_window)
            raw = np.mean(ratios)
            scores[i] = min(1.0, 1.0 / raw)
        return scores

    def objective(r):
        points = get_points(r)
        
        P = points
        E1 = np.roll(P, -1, axis=0) - P
        E2 = np.roll(P, -2, axis=0) - np.roll(P, -1, axis=0)
        cross = E1[:, 0] * E2[:, 1] - E1[:, 1] * E2[:, 0]
        
        penalty = np.sum(np.maximum(0, -cross)**2)
        
        scores = compute_scores(P)
        lam = 20.0
        soft_min = -np.log(np.sum(np.exp(-lam * scores))) / lam
        
        return -(soft_min) + 100.0 * penalty

    start_time = time.time()
    deadline = start_time + 580
    
    # Use COBYLA or Powell
    # Powell is robust
    try:
        result = minimize(objective, r0, method='Powell', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
        best_r = result.x
    except Exception:
        best_r = r0
        
    if time.time() > deadline:
        return get_points(best_r)
        
    # Polish
    try:
        result2 = minimize(objective, best_r, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
        best_r = result2.x
    except Exception:
        pass
        
    return get_points(best_r)
