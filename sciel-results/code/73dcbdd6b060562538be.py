import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _score(pts):
    # Vectorized min triangle area
    i_idx = np.arange(N)[:, np.newaxis, np.newaxis]
    j_idx = np.arange(N)[np.newaxis, :, np.newaxis]
    k_idx = np.arange(N)[np.newaxis, np.newaxis, :]
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    areas = 0.5 * np.abs(cross)
    
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_tri = np.min(valid_areas)
    
    # Hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        hull_area = 1.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return min_tri / hull_area

def _objective(flat_pts):
    pts = flat_pts.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    TIME_LIMIT = 110.0
    
    # Initializations
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_pts.flatten())
    
    # 2. Perturbed Regular 14-gon
    rng = np.random.RandomState(123)
    perturbed = reg_pts + rng.normal(0, 0.05, (N, 2))
    inits.append(perturbed.flatten())
    
    # 3. Random in unit disk
    for _ in range(2):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
        
    best_score = -np.inf
    best_pts = inits[0].reshape(N, 2)
    
    for init in inits:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        try:
            # Nelder-Mead
            res = minimize(_objective, init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(N, 2)
                
            # Optional: Powell refinement
            if time.time() - start_time < TIME_LIMIT - 20:
                res2 = minimize(_objective, res.x, method='Powell', 
                                options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-8})
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_pts = res2.x.reshape(N, 2)
                    
        except Exception:
            pass
            
    return best_pts
