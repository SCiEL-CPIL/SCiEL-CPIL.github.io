import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

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
    # Check for degeneracy
    if points.shape != (N, 2):
        return 0.0
    
    # Compute min triangle area
    min_area = float('inf')
    # Use local variables for speed
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        # Cross product magnitude
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    if min_area == 0.0:
        return 0.0
        
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-9:
        return 0.0
        
    return min_area / hull_area

def _get_points_2p(r, theta):
    # Outer heptagon
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles_outer)
    y_out = np.sin(angles_outer)
    
    # Inner heptagon
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
    # Hull is outer heptagon
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Optimize the 2-parameter model (concentric heptagons)
    # This provides a good starting point and a baseline.
    from scipy.optimize import differential_evolution
    
    bounds = [(0.1, 0.9), (0, 2*np.pi/7)]
    
    try:
        res_de = differential_evolution(
            lambda p: -_score_2p(p),
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-5
        )
        if res_de.success:
            p2 = res_de.x
            # Polish with Nelder-Mead
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

    # 2. General Local Optimization
    # We will try to improve the 2-param solution by allowing all points to move.
    # We will also try a few other starting configurations.
    
    starting_points = []
    
    if best_points is not None:
        starting_points.append(best_points)
        
    # Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    points_reg = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    starting_points.append(points_reg)
    
    # Two heptagons, r=0.5, theta=0
    points_h2 = _get_points_2p(0.5, 0.0)
    starting_points.append(points_h2)
    
    # Two heptagons, r=0.7, theta=pi/7
    points_h3 = _get_points_2p(0.7, np.pi/7)
    starting_points.append(points_h3)
    
    # Random
    np.random.seed(42)
    points_rand = np.random.rand(N, 2) * 2 - 1
    # Ensure non-degenerate? Random points are almost surely non-degenerate.
    starting_points.append(points_rand)
    
    for start_pts in starting_points:
        if time.time() > deadline:
            break
            
        # Flatten points for optimization
        x0 = start_pts.flatten()
        
        def neg_score_general(x):
            pts = x.reshape(N, 2)
            return -_score_general(pts)
        
        try:
            # Nelder-Mead is good for non-smooth problems
            res_gen = minimize(
                neg_score_general,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500}
            )
            
            if res_gen.success or (res_gen.fun < 0): # Check if we got a valid score
                score_gen = -res_gen.fun
                if score_gen > best_score:
                    best_score = score_gen
                    best_points = res_gen.x.reshape(N, 2)
                    
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.random.rand(N, 2)
        
    return best_points
