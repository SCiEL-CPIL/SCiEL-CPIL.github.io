import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14
# Precompute triple indices
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    
    # Check for degenerate cases: points too close
    # This helps avoid numerical issues
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-10:
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Extract points for all triples
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _score_two_ring(params, n_out, n_in):
    """
    Objective function for two-ring optimization.
    params: [r_in, theta_in]
    n_out: number of points on outer ring (radius 1.0)
    n_in: number of points on inner ring (radius r_in)
    """
    r_in, theta_in = params
    if r_in < 0.1 or r_in > 0.95:
        return 0.0
    
    # Generate points
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_in
    x_in = r_in * np.cos(angles_in)
    y_in = r_in * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _calculate_score(points)

def _optimize_two_ring(n_out, n_in, time_limit):
    """
    Optimize a two-ring configuration using differential evolution.
    Returns (best_score, best_points).
    """
    start_time = time.time()
    
    # Bounds for r_in and theta_in
    bounds = [(0.2, 0.9), (0.0, 2*np.pi)]
    
    try:
        # Differential evolution is good for global optimization in low dimensions
        # We limit the number of iterations to respect the time limit
        # maxiter is not strictly time-based, so we use a callback to check time
        def callback(xk, convergence):
            return time.time() - start_time > time_limit - 5
        
        res = differential_evolution(
            lambda x: -_score_two_ring(x, n_out, n_in),
            bounds,
            maxiter=200,
            tol=1e-6,
            callback=callback,
            seed=42,
            polish=True
        )
        
        if res.fun < 0:
            best_params = res.x
            best_score = -res.fun
            
            # Reconstruct points
            r_in, theta_in = best_params
            angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
            x_out = np.cos(angles_out)
            y_out = np.sin(angles_out)
            
            angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_in
            x_in = r_in * np.cos(angles_in)
            y_in = r_in * np.sin(angles_in)
            
            best_points = np.vstack([
                np.column_stack((x_out, y_out)),
                np.column_stack((x_in, y_in))
            ])
            
            return best_score, best_points
        else:
            return 0.0, None
            
    except Exception:
        return 0.0, None

def _generate_regular_polygon(r=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _generate_random_points(seed):
    rng = np.random.default_rng(seed)
    # Generate random points in unit disk
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _optimize_general(points, time_limit):
    """
    General local optimization using Nelder-Mead for any point configuration.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e9
        pts = x.reshape(N, 2)
        return -_calculate_score(pts)
        
    try:
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'disp': False}
        )
        
        if time.time() - start_time < time_limit:
            new_points = res.x.reshape(N, 2)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    except Exception:
        pass
        
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # 1. Baseline: Regular Polygon
    reg_points = _generate_regular_polygon(1.0)
    score, pts = _optimize_general(reg_points, 10.0)
    if score > best_score:
        best_score = score
        best_points = pts
    
    # 2. Structural Change: Two-Ring Optimization
    # Try different splits: 7+7, 6+8, 8+6
    splits = [(7, 7), (6, 8), (8, 6)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 30:
            break
            
        score, pts = _optimize_two_ring(n_out, n_in, 20.0)
        if score > best_score and pts is not None:
            best_score = score
            best_points = pts
            # Polish the result with general local optimization
            if time.time() < deadline - 10:
                pol_score, pol_pts = _optimize_general(pts, 10.0)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = pol_pts

    # 3. If time remains, try a 3-ring configuration (e.g., 5+5+4)
    # This is a more complex structural change.
    if time.time() < deadline - 20:
        # Define a 3-ring scorer
        def _score_three_ring(params):
            r1, r2, th1, th2 = params
            if r1 < 0.1 or r1 > 0.6 or r2 < 0.6 or r2 > 0.95:
                return 0.0
            
            n1, n2, n3 = 5, 5, 4
            
            a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
            a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + th1
            a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + th2
            
            pts1 = np.column_stack((np.cos(a1), np.sin(a1)))
            pts2 = np.column_stack((r1 * np.cos(a2), r1 * np.sin(a2)))
            pts3 = np.column_stack((r2 * np.cos(a3), r2 * np.sin(a3)))
            
            points = np.vstack([pts1, pts2, pts3])
            return _calculate_score(points)

        bounds = [(0.2, 0.6), (0.6, 0.9), (0, 2*np.pi), (0, 2*np.pi)]
        try:
            def callback(xk, convergence):
                return time.time() - start_time > deadline - 10
            res = differential_evolution(
                lambda x: -_score_three_ring(x),
                bounds,
                maxiter=100,
                tol=1e-5,
                callback=callback,
                seed=42,
                polish=True
            )
            if res.fun < 0:
                new_score = -res.fun
                if new_score > best_score:
                    # Reconstruct points for polish
                    r1, r2, th1, th2 = res.x
                    n1, n2, n3 = 5, 5, 4
                    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
                    a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + th1
                    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + th2
                    pts1 = np.column_stack((np.cos(a1), np.sin(a1)))
                    pts2 = np.column_stack((r1 * np.cos(a2), r1 * np.sin(a2)))
                    pts3 = np.column_stack((r2 * np.cos(a3), r2 * np.sin(a3)))
                    new_points = np.vstack([pts1, pts2, pts3])
                    
                    # Polish
                    pol_score, pol_pts = _optimize_general(new_points, 10.0)
                    if pol_score > best_score:
                        best_score = pol_score
                        best_points = pol_pts
        except Exception:
            pass

    # 4. Random Starts with Local Optimization (Fallback/Explorer)
    for seed in range(1, 10):
        if time.time() > deadline - 5:
            break
        start_pts = _generate_random_points(seed)
        score, pts = _optimize_general(start_pts, 5.0)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = reg_points
        
    return best_points
