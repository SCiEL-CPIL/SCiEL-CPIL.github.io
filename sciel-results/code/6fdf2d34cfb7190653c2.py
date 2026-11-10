import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 50

def _exact_covering_radius(points):
    """
    Calculate the exact covering radius in radians using ConvexHull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
        
    max_radius = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        d = np.abs(np.dot(normal, p0)) / norm_normal
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _smooth_covering_radius(points, beta=50.0):
    """
    Calculate a smooth approximation of the covering radius.
    Minimizing this is equivalent to maximizing the minimum distance to the origin
    among all hull facets, which minimizes the maximum circumradius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1e6
        
    ds = []
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            ds.append(0.0)
            continue
            
        d = np.abs(np.dot(normal, p0)) / norm_normal
        d = np.clip(d, 0.0, 1.0)
        ds.append(d)
        
    if not ds:
        return 1e6
        
    ds = np.array(ds)
    log_sum_exp = np.log(np.sum(np.exp(-beta * ds)))
    return (1.0 / beta) * log_sum_exp

def _repulsive_relaxation(points, steps=2000, alpha=0.1):
    """
    Relax points using repulsive forces to spread them out.
    """
    points = points.copy()
    for _ in range(steps):
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                forces[i] += diff / (dist_sq * np.sqrt(dist_sq))
        
        points += alpha * forces
        
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
    return points

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _exact_covering_radius(best_points)
    
    # 2. Repulsive Relaxation
    points = _repulsive_relaxation(points, steps=5000, alpha=0.05)
    current_radius = _exact_covering_radius(points)
    if current_radius < best_radius:
        best_points = points.copy()
        best_radius = current_radius
        
    # 3. Local Optimization with L-BFGS-B on Smooth Objective
    def spherical_to_cartesian(params):
        theta = params[0::2]
        phi = params[1::2]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
        
    def smooth_objective(params):
        if time.time() > deadline:
            return 1e6
        pts = spherical_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _smooth_covering_radius(pts, beta=50.0)
        
    def exact_objective(params):
        if time.time() > deadline:
            return 1e6
        pts = spherical_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _exact_covering_radius(pts)

    theta = np.arccos(np.clip(best_points[:, 2], -1.0, 1.0))
    phi = np.arctan2(best_points[:, 1], best_points[:, 0])
    params0 = np.zeros(2 * N)
    params0[0::2] = theta
    params0[1::2] = phi
    
    try:
        result = minimize(smooth_objective, params0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < 1e5:
            candidate_pts = spherical_to_cartesian(result.x)
            candidate_rad = _exact_covering_radius(candidate_pts)
            if candidate_rad < best_radius:
                best_points = candidate_pts
                best_radius = candidate_rad
                params0 = result.x
                
    except Exception:
        pass
        
    if time.time() > deadline:
        return best_points

    # 4. Polish with Nelder-Mead on Exact Objective
    try:
        result2 = minimize(exact_objective, params0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result2.fun < best_radius:
            best_points = spherical_to_cartesian(result2.x)
            best_radius = result2.fun
            
    except Exception:
        pass

    # 5. Random Restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for _ in range(3):
            if time.time() > deadline:
                break
            
            noise = np.random.normal(0, 0.05, size=best_points.shape)
            perturbed = best_points + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed /= norms
            
            theta_p = np.arccos(np.clip(perturbed[:, 2], -1.0, 1.0))
            phi_p = np.arctan2(perturbed[:, 1], perturbed[:, 0])
            params_p = np.zeros(2 * N)
            params_p[0::2] = theta_p
            params_p[1::2] = phi_p
            
            try:
                res_p = minimize(smooth_objective, params_p, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-8})
                if res_p.fun < 1e5:
                    pts_p = spherical_to_cartesian(res_p.x)
                    rad_p = _exact_covering_radius(pts_p)
                    if rad_p < best_radius:
                        best_points = pts_p
                        best_radius = rad_p
                        
                        res_polish = minimize(exact_objective, res_p.x, method='Nelder-Mead', 
                                             options={'maxiter': 200, 'xatol': 1e-8})
                        if res_polish.fun < best_radius:
                            best_points = spherical_to_cartesian(res_polish.x)
                            best_radius = res_polish.fun
            except Exception:
                pass

    return best_points
