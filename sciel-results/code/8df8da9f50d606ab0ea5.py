import numpy as np
import time
import itertools
import scipy.optimize as opt
import scipy.spatial as spat

N = 13
DEADLINE = 150.0 # Seconds
START_TIME = time.time()

# Precompute triangle indices
TRI_INDICES = list(itertools.combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute minimum triangle area
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def points_from_fourier(params):
    """
    Generate 13 points from 4 Fourier parameters:
    a1, phi1, a2, phi2
    Theta_k = 2*pi*k/13 + a1*cos(2*pi*k/13 + phi1) + a2*cos(4*pi*k/13 + phi2)
    Radii are fixed to 1.
    """
    a1, phi1, a2, phi2 = params
    k = np.arange(N)
    base_angles = 2 * np.pi * k / N
    # First harmonic
    pert1 = a1 * np.cos(base_angles + phi1)
    # Second harmonic
    pert2 = a2 * np.cos(2 * base_angles + phi2)
    
    angles = base_angles + pert1 + pert2
    
    # Ensure angles are sorted to maintain convexity order?
    # With small a1, a2, the order k=0..12 should be preserved.
    # If perturbations are too large, points might cross.
    # Bounds on a1, a2 will prevent this.
    
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack([x, y])

def objective_fourier(params):
    """
    Objective function for the optimizer.
    We want to MAXIMIZE calculate_score, so we MINIMIZE -calculate_score.
    """
    pts = points_from_fourier(params)
    score = calculate_score(pts)
    return -score

def construct():
    global START_TIME
    current_time = time.time()
    elapsed = current_time - START_TIME
    time_left = DEADLINE - elapsed
    
    if time_left < 5:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    # Parameters: a1, phi1, a2, phi2
    # a1, a2 are amplitudes in radians. 
    # Max perturbation should not exceed ~pi/13 to avoid crossing neighbors?
    # Derivative of angle w.r.t k is 2pi/13. Perturbation derivative is a1 * 2pi/13 * sin(...).
    # To keep dTheta/dk > 0, we need a1 * 2pi/13 * 1 < 2pi/13 => a1 < 1.
    # So bounds [-0.5, 0.5] are safe.
    bounds = [(-0.5, 0.5), (0, 2*np.pi), (-0.5, 0.5), (0, 2*np.pi)]
    
    best_score = -np.inf
    best_pts = None
    best_params = None
    
    # 1. Global Search with Differential Evolution
    # DE is robust and good for low-dimensional global problems.
    try:
        # maxiter is number of generations. popsize is multiplier.
        # 4D problem, small bounds.
        # We limit time by checking after each generation? DE doesn't support callback easily for stopping mid-generation in older scipy, 
        # but we can just run a moderate number of iterations.
        # 100 generations * 20 individuals * 4D evals is fast.
        res = opt.differential_evolution(
            objective_fourier, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-4,
            polish=True # This will run Nelder-Mead on the best result
        )
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
            best_pts = points_from_fourier(best_params)
            
    except Exception:
        pass

    # 2. Local Polish with Powell on the best params found
    if best_params is not None:
        try:
            # Use Powell for higher precision on the 4D space
            res2 = opt.minimize(
                objective_fourier, 
                best_params, 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            
            score2 = -res2.fun
            if score2 > best_score:
                best_score = score2
                best_params = res2.x
                best_pts = points_from_fourier(best_params)
                
        except Exception:
            pass

    # 3. Try a few random restarts if time permits
    current_time = time.time()
    elapsed = current_time - START_TIME
    time_left = DEADLINE - elapsed
    
    rng = np.random.default_rng(42)
    
    while time_left > 20:
        # Random init in bounds
        rand_params = np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(0, 2*np.pi),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0, 2*np.pi)
        ])
        
        try:
            res3 = opt.minimize(
                objective_fourier, 
                rand_params, 
                method='Powell', 
                options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            
            score3 = -res3.fun
            if score3 > best_score:
                best_score = score3
                best_params = res3.x
                best_pts = points_from_fourier(best_params)
                
        except Exception:
            pass
            
        current_time = time.time()
        elapsed = current_time - START_TIME
        time_left = DEADLINE - elapsed

    if best_pts is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
