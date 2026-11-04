import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize, differential_evolution

N = 14

# Precompute triple indices
TRIPLE_INDICES = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            TRIPLE_INDICES.append((i, j, k))
TRIPLE_INDICES = np.array(TRIPLE_INDICES)

def _score_general(points):
    """
    Calculate score for general N points.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    if min_area == 0.0:
        return 0.0
        
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-9:
        return 0.0
        
    return min_area / hull_area

def _get_points_2p(r, theta):
    """Two concentric heptagons."""
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles_outer)
    y_out = np.sin(angles_outer)
    
    angles_inner = angles_outer + theta
    x_in = r * np.cos(angles_inner)
    y_in = r * np.sin(angles_inner)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _score_2p(params):
    r, theta = params
    if r <= 0 or r >= 1:
        return 0.0
    points = _get_points_2p(r, theta)
    hull_area = 0.5 * 7 * np.sin(2*np.pi/7)
    
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

def _get_points_3ring(r1, r2, r3, theta1, theta2, theta3):
    """
    Three concentric rings: 5 points on outer, 5 on middle, 4 on inner.
    Radii: r1 (outer=1), r2 (middle), r3 (inner)
    Angular offsets: theta1, theta2, theta3
    """
    # Outer ring: 5 points at radius 1
    angles_o = np.linspace(0, 2*np.pi, 5, endpoint=False) + theta1
    x_o = np.cos(angles_o)
    y_o = np.sin(angles_o)
    
    # Middle ring: 5 points at radius r2
    angles_m = np.linspace(0, 2*np.pi, 5, endpoint=False) + theta2
    x_m = r2 * np.cos(angles_m)
    y_m = r2 * np.sin(angles_m)
    
    # Inner ring: 4 points at radius r3
    angles_i = np.linspace(0, 2*np.pi, 4, endpoint=False) + theta3
    x_i = r3 * np.cos(angles_i)
    y_i = r3 * np.sin(angles_i)
    
    points = np.vstack([
        np.column_stack((x_o, y_o)),
        np.column_stack((x_m, y_m)),
        np.column_stack((x_i, y_i))
    ])
    return points

def _score_3ring(params):
    r2, r3, theta1, theta2, theta3 = params
    # r1 is fixed at 1.0 (outer ring)
    if r2 <= 0 or r2 >= 1 or r3 <= 0 or r3 >= 1:
        return 0.0
    # Ensure r2 > r3 for proper nesting
    if r2 <= r3:
        return 0.0
    
    points = _get_points_3ring(1.0, r2, r3, theta1, theta2, theta3)
    
    # Convex hull is the outer pentagon
    hull_area = 0.5 * 5 * np.sin(2*np.pi/5)
    
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

def _get_points_3ring_alt(r1, r2, r3, theta1, theta2, theta3):
    """
    Alternative 3-ring: 6 points outer, 4 middle, 4 inner.
    """
    # Outer ring: 6 points at radius 1
    angles_o = np.linspace(0, 2*np.pi, 6, endpoint=False) + theta1
    x_o = np.cos(angles_o)
    y_o = np.sin(angles_o)
    
    # Middle ring: 4 points at radius r2
    angles_m = np.linspace(0, 2*np.pi, 4, endpoint=False) + theta2
    x_m = r2 * np.cos(angles_m)
    y_m = r2 * np.sin(angles_m)
    
    # Inner ring: 4 points at radius r3
    angles_i = np.linspace(0, 2*np.pi, 4, endpoint=False) + theta3
    x_i = r3 * np.cos(angles_i)
    y_i = r3 * np.sin(angles_i)
    
    points = np.vstack([
        np.column_stack((x_o, y_o)),
        np.column_stack((x_m, y_m)),
        np.column_stack((x_i, y_i))
    ])
    return points

def _score_3ring_alt(params):
    r2, r3, theta1, theta2, theta3 = params
    if r2 <= 0 or r2 >= 1 or r3 <= 0 or r3 >= 1:
        return 0.0
    if r2 <= r3:
        return 0.0
    
    points = _get_points_3ring_alt(1.0, r2, r3, theta1, theta2, theta3)
    
    # Convex hull is the outer hexagon
    hull_area = 0.5 * 6 * np.sin(2*np.pi/6)
    
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_points = None
    
    # 1. Optimize 2-ring model
    bounds_2p = [(0.1, 0.9), (0, 2*np.pi/7)]
    try:
        res_de = differential_evolution(
            lambda p: -_score_2p(p),
            bounds_2p,
            seed=42,
            maxiter=80,
            popsize=15,
            tol=1e-5
        )
        if res_de.success:
            p2 = res_de.x
            res_nm = minimize(
                lambda p: -_score_2p(p),
                p2,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
            )
            score_2p = -res_nm.fun
            points_2p = _get_points_2p(res_nm.x[0], res_nm.x[1])
            if score_2p > best_score:
                best_score = score_2p
                best_points = points_2p
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return np.random.rand(N, 2)
    
    # 2. Optimize 3-ring model (5-5-4)
    bounds_3p = [(0.1, 0.95), (0.01, 0.95), (0, 2*np.pi/5), (0, 2*np.pi/5), (0, 2*np.pi/4)]
    try:
        res_de3 = differential_evolution(
            lambda p: -_score_3ring(p),
            bounds_3p,
            seed=123,
            maxiter=100,
            popsize=18,
            tol=1e-5
        )
        if res_de3.success:
            p3 = res_de3.x
            res_nm3 = minimize(
                lambda p: -_score_3ring(p),
                p3,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300}
            )
            score_3p = -res_nm3.fun
            points_3p = _get_points_3ring(1.0, res_nm3.x[0], res_nm3.x[1], 
                                            res_nm3.x[2], res_nm3.x[3], res_nm3.x[4])
            if score_3p > best_score:
                best_score = score_3p
                best_points = points_3p
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return np.random.rand(N, 2)
    
    # 3. Optimize 3-ring alt model (6-4-4)
    bounds_3p_alt = [(0.1, 0.95), (0.01, 0.95), (0, 2*np.pi/6), (0, 2*np.pi/4), (0, 2*np.pi/4)]
    try:
        res_de3a = differential_evolution(
            lambda p: -_score_3ring_alt(p),
            bounds_3p_alt,
            seed=456,
            maxiter=100,
            popsize=18,
            tol=1e-5
        )
        if res_de3a.success:
            p3a = res_de3a.x
            res_nm3a = minimize(
                lambda p: -_score_3ring_alt(p),
                p3a,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300}
            )
            score_3pa = -res_nm3a.fun
            points_3pa = _get_points_3ring_alt(1.0, res_nm3a.x[0], res_nm3a.x[1],
                                                res_nm3a.x[2], res_nm3a.x[3], res_nm3a.x[4])
            if score_3pa > best_score:
                best_score = score_3pa
                best_points = points_3pa
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return np.random.rand(N, 2)
    
    # 4. General local optimization from best structured solution
    if best_points is not None:
        x0 = best_points.flatten()
        
        def neg_score_general(x):
            pts = x.reshape(N, 2)
            return -_score_general(pts)
        
        try:
            res_gen = minimize(
                neg_score_general,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 800}
            )
            score_gen = -res_gen.fun
            if score_gen > best_score:
                best_score = score_gen
                best_points = res_gen.x.reshape(N, 2)
        except Exception:
            pass
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
