import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 140.0  # Safety margin before 150s kill

def _hull_area(pts):
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """
    Vectorized calculation of all triangle areas.
    Returns an array of shape (N*(N-1)*(N-2)/6, )
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorized cross product for k > j
            # Triangle (i, j, k)
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas.extend(0.5 * np.abs(cross))
            
    return np.array(areas)

def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 0.0
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    return min_a / hull_a

def _objective(x, beta):
    """
    Objective function to minimize: hull_area / soft_min(areas, beta)
    Minimizing this is equivalent to maximizing soft_min(areas) / hull_area.
    """
    points = x.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 1e10
        
    min_a = np.min(areas)
    if min_a < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    # Soft min calculation
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability:
    # soft_min = min_val - (1/beta) * log(sum(exp(-beta * (areas - min_val))))
    diff = areas - min_a
    # Avoid underflow/overflow in exp
    # beta * diff is >= 0, so exp(-beta * diff) is in (0, 1]
    exp_terms = np.exp(-beta * diff)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_a - log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
        
    return hull_area / soft_min_val

def _generate_initials():
    """
    Generate a diverse set of initial configurations.
    """
    inits = []
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p_reg)
    
    # 2. Concentric rings: 7 inner, 7 outer
    # Inner
    ang_in = 2 * np.pi * np.arange(7) / 7
    p_in = np.column_stack([0.5 * np.cos(ang_in), 0.5 * np.sin(ang_in)])
    # Outer
    ang_out = 2 * np.pi * np.arange(7) / 7 + np.pi / 7 # Offset
    p_out = np.column_stack([1.0 * np.cos(ang_out), 1.0 * np.sin(ang_out)])
    p_rings = np.vstack([p_in, p_out])
    inits.append(p_rings)
    
    # 3. Concentric rings: 6 inner, 8 outer (to test asymmetry)
    ang_in6 = 2 * np.pi * np.arange(6) / 6
    p_in6 = np.column_stack([0.6 * np.cos(ang_in6), 0.6 * np.sin(ang_in6)])
    ang_out8 = 2 * np.pi * np.arange(8) / 8
    p_out8 = np.column_stack([1.0 * np.cos(ang_out8), 1.0 * np.sin(ang_out8)])
    p_rings2 = np.vstack([p_in6, p_out8])
    inits.append(p_rings2)
    
    # 4. Random points on a circle (various seeds)
    rng = np.random.RandomState(42)
    for seed in range(10):
        r = np.random.RandomState(seed)
        angles = r.uniform(0, 2 * np.pi, N)
        angles.sort() # Sort to maintain convexity order roughly
        p_rand = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(p_rand)
        
    # 5. Perturbed regular polygon
    for seed in range(10):
        r = np.random.RandomState(seed)
        noise = r.normal(0, 0.1, size=(N, 2))
        p_noisy = p_reg + noise
        inits.append(p_noisy)
        
    # 6. Random points in a disk
    for seed in range(10):
        r = np.random.RandomState(seed + 100)
        angles = r.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(r.uniform(0, 1, N)) # Uniform area distribution
        p_disk = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(p_disk)
        
    return inits

def construct():
    start_time = time.time()
    
    inits = _generate_initials()
    
    best_score = -np.inf
    best_pts = inits[0]
    
    # Evaluate initial scores to prioritize
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
        if s > best_score:
            best_score = s
            best_pts = init
            
    # Sort by score, best first
    sorted_indices = np.argsort(init_scores)[::-1]
    sorted_inits = [inits[i] for i in sorted_indices]
    
    # Optimization strategy:
    # For each of the top K initial configurations, run a multi-stage L-BFGS-B optimization.
    # Stage 1: Low beta (smooth landscape)
    # Stage 2: Medium beta
    # Stage 3: High beta (sharp min)
    
    top_k = 20
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for i in range(min(top_k, len(sorted_inits))):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        x0 = sorted_inits[i].flatten()
        
        # Check if we have enough time for this candidate
        if time.time() - start_time > DEADLINE - 10:
            break
            
        current_x = x0.copy()
        current_obj = _objective(current_x, betas[0])
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 2:
                break
                
            def obj_func(x, b=beta):
                return _objective(x, b)
                
            try:
                res = minimize(obj_func, current_x, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500})
                
                if res.fun < current_obj:
                    current_obj = res.fun
                    current_x = res.x
            except Exception:
                pass
        
        # Final evaluation with exact score
        final_pts = current_x.reshape(-1, 2)
        final_score = _score(final_pts)
        
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Normalize to unit area for return
    hull_a = _hull_area(best_pts)
    if hull_a > 1e-10:
        best_pts = best_pts / np.sqrt(hull_a)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
