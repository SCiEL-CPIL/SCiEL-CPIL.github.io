import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas
    # There are N*(N-1)*(N-2)/6 triangles
    n = len(points)
    min_area = float('inf')
    
    # Optimization: Use vectorized calculation if possible, but N=14 is small
    # Loop is fine: 364 triangles
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle with vertices p_i, p_j, p_k
                # 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
                        
    return min_area / hull_area

def neg_score(x):
    """
    Objective function for minimization (we maximize score).
    x is a flat array of length 2*N.
    """
    points = x.reshape((N, 2))
    score = compute_score(points)
    return -score

def _three_ring_points(params):
    """
    Generate 14 points using 3 rings: 7 outer, 4 middle, 3 inner.
    params: [r_mid, r_inner, theta_mid, theta_inner]
    Outer ring is fixed at radius 1.0 with 7 points.
    Middle ring has 4 points at radius r_mid, rotated by theta_mid.
    Inner ring has 3 points at radius r_inner, rotated by theta_inner.
    """
    r_mid, r_inner, theta_mid, theta_inner = params
    
    # Outer ring: 7 points at radius 1.0
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    
    # Middle ring: 4 points at radius r_mid
    angles_mid = np.linspace(0, 2 * np.pi, 4, endpoint=False) + theta_mid
    p_mid = np.column_stack((r_mid * np.cos(angles_mid), r_mid * np.sin(angles_mid)))
    
    # Inner ring: 3 points at radius r_inner
    angles_inner = np.linspace(0, 2 * np.pi, 3, endpoint=False) + theta_inner
    p_inner = np.column_stack((r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)))
    
    points = np.vstack((p_outer, p_mid, p_inner))
    return points

def _neg_three_ring_score(params):
    """
    Objective function for the 3-ring parameterization.
    params: [r_mid, r_inner, theta_mid, theta_inner]
    """
    r_mid, r_inner, theta_mid, theta_inner = params
    
    if r_mid < 0.0 or r_mid > 1.0 or r_inner < 0.0 or r_inner > 1.0:
        return 1e9
    
    points = _three_ring_points(params)
    score = compute_score(points)
    return -score

def _two_ring_points(r, theta):
    """
    Generate 14 points: 7 on outer ring (radius 1.0), 7 on inner ring (radius r).
    Inner ring is rotated by theta.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + theta
    
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    p_inner = np.column_stack((r * np.cos(angles_inner), r * np.sin(angles_inner)))
    
    points = np.vstack((p_outer, p_inner))
    return points

def _neg_ring_score(params):
    """
    Objective function for the 2-ring parameterization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.0 or r > 1.0:
        return 1e9
    points = _two_ring_points(r, theta)
    score = compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    
    # 1. Global search over 3-ring family: [r_mid, r_inner, theta_mid, theta_inner]
    # r_mid, r_inner in [0, 1], theta_mid in [0, 2*pi/4], theta_inner in [0, 2*pi/3]
    bounds_3ring = [(0.0, 1.0), (0.0, 1.0), (0.0, np.pi/2), (0.0, 2*np.pi/3)]
    
    try:
        res_de_3 = differential_evolution(
            _neg_three_ring_score, 
            bounds_3ring, 
            seed=42, 
            maxiter=150, 
            tol=1e-8,
            polish=True,
            init='sobol',
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.9
        )
        
        if res_de_3.success or res_de_3.fun < 0:
            points_3 = _three_ring_points(res_de_3.x)
            score_3 = compute_score(points_3)
            
            if score_3 > best_score:
                best_score = score_3
                best_points = points_3.copy()
    except Exception:
        pass

    # 2. Also search 2-ring family as backup
    if time.time() - start_time < deadline - 30:
        bounds_2ring = [(0.0, 1.0), (0.0, 2 * np.pi / 7)]
        try:
            res_de_2 = differential_evolution(
                _neg_ring_score, 
                bounds_2ring, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=True,
                init='sobol',
                popsize=15
            )
            
            if res_de_2.success or res_de_2.fun < 0:
                points_2 = _two_ring_points(res_de_2.x[0], res_de_2.x[1])
                score_2 = compute_score(points_2)
                
                if score_2 > best_score:
                    best_score = score_2
                    best_points = points_2.copy()
        except Exception:
            pass

    # Fallback: coarse grid if nothing found
    if best_points is None:
        r_vals = np.linspace(0.2, 0.9, 8)
        th_vals = np.linspace(0, 2*np.pi/7, 8)
        for r in r_vals:
            for th in th_vals:
                points = _two_ring_points(r, th)
                s = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
    
    # 3. Local refinement in full 28D space from the best symmetric start
    if best_points is not None:
        x0 = best_points.flatten()
        
        # Use Nelder-Mead for local refinement
        try:
            res_nm = minimize(neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            new_points = res_nm.x.reshape((N, 2))
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass
        
        # Try Powell as well
        if time.time() - start_time < deadline - 10:
            try:
                res_pw = minimize(neg_score, x0, method='Powell', 
                            options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
                new_points = res_pw.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            except Exception:
                pass

    # 4. Try a few random starts with local optimization
    if time.time() - start_time < deadline - 15:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() - start_time > deadline - 10:
                break
            # Random points in unit disk
            r = np.sqrt(rng.random(N))
            theta = rng.random(N) * 2 * np.pi
            p_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            
            try:
                res_r = minimize(neg_score, p_rand.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10})
                new_points = res_r.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            except Exception:
                pass

    return best_points
