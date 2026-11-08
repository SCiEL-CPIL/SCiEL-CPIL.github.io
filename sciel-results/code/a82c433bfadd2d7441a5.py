"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate min triangle area efficiently
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        diffs = points[i+1:] - pi
        m = len(diffs)
        if m < 2:
            continue
        
        # Compute cross products for all pairs in diffs
        # cross[a, b] = det(diffs[a], diffs[b])
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        
        # We only care about a < b
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        if not np.any(mask):
            continue
            
        areas = 0.5 * np.abs(cross[mask])
        current_min = np.min(areas)
        if current_min < min_area:
            min_area = current_min
            
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate using soft-min for the minimum triangle area.
    This makes the objective differentiable, allowing gradient-based optimization.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate all triangle areas
    areas = []
    for i in range(n - 2):
        pi = points[i]
        diffs = points[i+1:] - pi
        m = len(diffs)
        if m < 2:
            continue
        
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        if not np.any(mask):
            continue
            
        areas.append(0.5 * np.abs(cross[mask]))
    
    if not areas:
        return 0.0
        
    all_areas = np.concatenate(areas)
    
    # Soft-min approximation
    min_val = np.min(all_areas)
    max_val = np.max(all_areas)
    if max_val - min_val < 1e-10:
        soft_min = min_val
    else:
        # Use log-sum-exp trick for numerical stability
        shifted = all_areas - max_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) - np.log(len(all_areas))
        soft_min = max_val + log_sum / (-beta)
        
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return soft_min / hull_area

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_soft_min_score(points, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _score(x0)
    best_points = x0.copy()
    
    # List of initial configurations to try
    configs = [
        x0,
        x0 + 0.1 * np.random.randn(N, 2),
        x0 + 0.05 * np.random.randn(N, 2),
        # Two rings: 7 outer, 7 inner
        np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                             np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))]),
            np.column_stack([0.5*np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7), 
                             0.5*np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)])
        ]),
        # Ellipse
        np.column_stack([1.2 * np.cos(angles), 0.8 * np.sin(angles)])
    ]
    
    # Phase 1: Nelder-Mead on the exact (non-smooth) objective
    for x_init in configs:
        if time.time() > deadline - 40.0:
            break
            
        try:
            res = minimize(_neg_score, x_init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if time.time() > deadline - 40.0:
                break
                
            p_opt = res.x.reshape(-1, 2)
            s_opt = _score(p_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_points = p_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Polish with BFGS on the smooth surrogate
    if time.time() < deadline - 20.0:
        try:
            # Use the best point found so far as initial guess for polishing
            x_polish_init = best_points.flatten().copy()
            
            # Use a moderate beta for the soft-min to balance smoothness and accuracy
            res_polish = minimize(
                lambda x: _neg_soft_score(x, beta=50.0), 
                x_polish_init, 
                method='BFGS',
                options={'maxiter': 1000, 'gtol': 1e-8, 'disp': False}
            )
            
            p_polished = res_polish.x.reshape(-1, 2)
            s_polished = _score(p_polished)
            
            if s_polished > best_score:
                best_score = s_polished
                best_points = p_polished.copy()
        except Exception:
            pass
            
    # Phase 3: Final high-precision Nelder-Mead polish on the exact objective
    if time.time() < deadline - 5.0:
        try:
            x_final_init = best_points.flatten().copy()
            
            res_final = minimize(
                _neg_score, 
                x_final_init, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            p_final = res_final.x.reshape(-1, 2)
            s_final = _score(p_final)
            
            if s_final > best_score:
                best_score = s_final
                best_points = p_final.copy()
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
