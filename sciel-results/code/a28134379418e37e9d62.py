import numpy as np
import time

def _min_enclosing_hex_side(centers, theta_outer):
    """Find minimum side length of a regular hexagon (rotation theta_outer rad, center origin)
    that contains all unit hexagons at given centers (all rotated 30 deg = pi/6 rad)."""
    phi = theta_outer
    outer_normals = phi + np.pi/6 + np.arange(6) * np.arange(6) * 0 + np.arange(6) * np.pi/3
    
    L_min = 0.0
    for alpha in outer_normals:
        u = np.array([np.cos(alpha), np.sin(alpha)])
        max_inner_support = 0.0
        for c in centers:
            h_i = (np.sqrt(3)/2) * np.cos(alpha - np.pi/3) + c[0]*u[0] + c[1]*u[1]
            if h_i > max_inner_support:
                max_inner_support = h_i
        L_candidate = max_inner_support / (np.sqrt(3)/2)
        if L_candidate > L_min:
            L_min = L_candidate
    
    return L_min

def _overlap_check(centers, eps=1e-9):
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < np.sqrt(3) - eps:
                return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    def initial_centers():
        centers = []
        centers.append((0.0, 0.0))
        for k in range(6):
            angle = np.deg2rad(k * 60)
            centers.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
        for k in range(4):
            angle = np.deg2rad(30 + k * 60)
            centers.append((3.0 * np.cos(angle), 3.0 * np.sin(angle)))
        return np.array(centers)
    
    centers_init = initial_centers()
    
    def objective(params):
        c = params[:22].reshape(11, 2)
        theta = params[22]
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                d = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d)**2
        L = _min_enclosing_hex_side(c, theta)
        return L + 100.0 * penalty
    
    from scipy.optimize import minimize
    
    x0 = np.concatenate([centers_init.flatten(), [np.pi/6]])
    
    best_params = x0.copy()
    best_obj = objective(x0)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
            if result2.fun < best_obj:
                best_params = result2.x
                best_obj = result2.fun
        except Exception:
            pass
    
    c_best = best_params[:22].reshape(11, 2)
    theta_best = best_params[22]
    
    if not _overlap_check(c_best):
        c_best = centers_init
        theta_best = np.pi/6
    
    L_final = _min_enclosing_hex_side(c_best, theta_best)
    
    inner = []
    for i in range(11):
        inner.append([float(c_best[i,0]), float(c_best[i,1]), 30.0])
    
    outer_center = (0.0, 0.0)
    side = float(L_final)
    angle_deg = float(np.degrees(theta_best))
    
    return inner, outer_center, side, angle_deg
