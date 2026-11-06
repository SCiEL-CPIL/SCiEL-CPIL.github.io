"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi, exp, log, sqrt

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    # We need product over all ordered pairs (i,j) with i != j
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_score(points):
    return -_score(points)

def _params_to_points_d4(params):
    """
    Parameterize 8 points with D4 symmetry (square symmetry).
    This reduces the search space significantly and enforces a structure
    likely to be optimal or near-optimal.
    
    Points are generated from 2 unique points (x1, y1) and (x2, y2) 
    assuming the first point is on the x-axis (due to rotational symmetry)
    and the second is in the first quadrant.
    
    Actually, let's use a simpler D4 parameterization:
    4 points on the axes? No, that's too restrictive.
    
    Let's use:
    Point 0: (r1, 0)
    Point 1: (r2*cos(a2), r2*sin(a2))
    Point 2: (0, r3)  <- Wait, D4 symmetry of a square means if (x,y) is a point,
                          (-y,x), (-x,-y), (y,-x) are also points.
    So we only need 2 unique points: (x1, y1) and (x2, y2)?
    No, D4 orbit size is 4 unless on symmetry axes.
    
    Let's assume the optimal configuration preserves the cyclic order of the regular octagon
    but allows for radial and angular distortions that break full D4 but keep D2 (180 deg rotational symmetry)?
    
    Let's stick to a robust global search on Cartesian coordinates first, 
    and use a specific structured family for the "deterministic" part.
    
    Structured Family: "Squashed Octagon"
    Points: 
    (1, 0), (s, s), (0, 1), (-s, s), (-1, 0), (-s, -s), (0, -1), (s, -s)
    This is a D4 symmetric configuration.
    Parameter s controls the "squash".
    Regular octagon corresponds to s = tan(pi/8) = sqrt(2)-1 ≈ 0.414.
    """
    s = params[0]
    # Ensure s is positive and reasonable
    s = max(0.01, min(1.0, s))
    
    pts = np.array([
        [1.0, 0.0],
        [s, s],
        [0.0, 1.0],
        [-s, s],
        [-1.0, 0.0],
        [-s, -s],
        [0.0, -1.0],
        [s, -s]
    ])
    return pts

def _score_d4(params):
    pts = _params_to_points_d4(params)
    return _score(pts)

def _neg_score_d4(params):
    return -_score_d4(params)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    best_score = 0.0
    best_points = None
    
    # 1. Initialize with regular octagon
    regular_octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    current_score = _score(regular_octagon)
    if current_score > best_score:
        best_score = current_score
        best_points = regular_octagon.copy()
    
    # 2. Optimize the D4 symmetric "squashed octagon" family
    # This is a 1D optimization problem, very fast and robust.
    try:
        # s ranges from 0.1 to 1.0
        # Regular octagon s = sqrt(2)-1
        x0_d4 = np.array([sqrt(2) - 1])
        
        # Use grid search + local refine for the 1D case
        s_vals = np.linspace(0.1, 1.0, 100)
        best_s = x0_d4[0]
        best_s_score = current_score
        
        for s in s_vals:
            sc = _score_d4(np.array([s]))
            if sc > best_s_score:
                best_s_score = sc
                best_s = s
        
        # Refine around best_s
        if time.time() < deadline - 5.0:
            result = minimize(_neg_score_d4, np.array([best_s]), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.success:
                sc = _score_d4(result.x)
                if sc > best_s_score:
                    best_s_score = sc
                    best_s = result.x[0]
        
        if best_s_score > best_score:
            best_score = best_s_score
            best_points = _params_to_points_d4(np.array([best_s]))
    except Exception:
        pass

    # 3. Global Search: Simulated Annealing on Cartesian Coordinates
    # This allows exploring configurations that break D4 symmetry.
    
    # We work in log-space of distances? No, direct coordinates.
    # To handle scaling invariance, we can fix the first point at (0,0) and the second at (1,0)?
    # No, that breaks rotational symmetry and might bias the search.
    # Better: Fix the center of mass at (0,0) and fix the scale by fixing the max distance to 2?
    # Scaling is handled in the score function, so we can search in arbitrary scale.
    # However, to avoid drift, we can re-center and re-scale during the search steps.
    
    rng = np.random.RandomState(42)
    
    # Start from the best found so far
    current_pts = best_points.copy()
    current_score = best_score
    
    # Temperature schedule
    # We want to explore for a while, then cool down.
    T = 0.05  # Initial temperature (score is around 1.0, so this is a small perturbation)
    T_min = 1e-6
    cooling = 0.999
    
    max_iterations = 50000
    # Estimate time per iteration
    # Score calculation is O(n^2) = 64 ops, very fast.
    # 50000 iterations should take < 5 seconds.
    
    for i in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a perturbation
        # Random walk in Cartesian space
        step_size = 0.1
        noise = rng.randn(8, 2) * step_size
        new_pts = current_pts + noise
        
        # Optional: Re-center to keep points in a reasonable range
        # new_pts -= np.mean(new_pts, axis=0)
        
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score or rng.random() < exp((new_score - current_score) / T):
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Adaptive step size: if we're stuck, increase step size slightly?
        # Or if we're improving, decrease it?
        # Let's keep it simple for now.
        
        # Every 1000 iterations, try a larger jump to escape local minima
        if i % 1000 == 999:
            jump_size = 0.5
            noise = rng.randn(8, 2) * jump_size
            new_pts = current_pts + noise
            new_score = _score(new_pts)
            if new_score > current_score or rng.random() < exp((new_score - current_score) / T):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()

    # 4. Local Refinement of the best point found by SA
    if time.time() < deadline - 5.0:
        # Flatten points to 1D array for minimize
        x0_flat = best_points.flatten()
        
        def _score_flat(x):
            pts = x.reshape(8, 2)
            return _score(pts)
        
        def _neg_score_flat(x):
            return -_score_flat(x)
        
        try:
            result = minimize(_neg_score_flat, x0_flat, method='Nelder-Mead',
                              options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if result.success:
                candidate_pts = result.x.reshape(8, 2)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts
        except Exception:
            pass

    # 5. Try a few random multi-starts with local optimization if time permits
    if time.time() < deadline - 10.0:
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Random start near the best point
            perturbation = rng.randn(8, 2) * 0.5
            x_start = best_points + perturbation
            
            try:
                result = minimize(_neg_score_flat, x_start.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                
                if result.success:
                    candidate_pts = result.x.reshape(8, 2)
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_pts
            except Exception:
                continue

    # Final check: Ensure distinctness and bounds
    if best_points is None:
        best_points = regular_octagon
    
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        # Fallback to regular octagon if points collapsed
        best_points = regular_octagon
    
    # Ensure coordinates are within bounds
    # |coordinate| < 10^6 is easily satisfied
    
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
