import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(points):
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return np.exp(log_score)

def _two_square_points(params):
    """
    Generate 8 points from two concentric squares.
    params: [r1, r2, theta1, theta2]
    Square 1: radius r1, rotation theta1
    Square 2: radius r2, rotation theta2
    """
    r1, r2, theta1, theta2 = params
    # Square 1 angles: theta1, theta1 + pi/2, theta1 + pi, theta1 + 3*pi/2
    angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2 angles: theta2, theta2 + pi/2, theta2 + pi, theta2 + 3*pi/2
    angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    pts = np.vstack([pts1, pts2])
    return pts

def _neg_log_score_two_square(params):
    """
    Compute negative log score for the two-square configuration.
    """
    pts = _two_square_points(params)
    n = 8
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return 1e10
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return -log_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Baseline: Regular Octagon
    # Regular octagon is two squares with r1=r2=1, theta1=0, theta2=pi/8
    # Or just generate directly
    angles = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = octagon.copy()
    best_score = _score(octagon)
    
    # Parameters: r1, r2, theta1, theta2
    # r1, r2 > 0
    # theta1, theta2 in [0, 2*pi)
    # Note: Due to symmetry, we can fix theta1=0 and search theta2 in [0, pi/4]
    # But let's allow full range for safety, or reduce dimensionality.
    # Fixing theta1=0 reduces symmetry breaking. Let's fix theta1=0.
    # Then we have 3 params: r1, r2, theta2.
    # Actually, let's keep 4 params but constrain theta1 to 0 in bounds? 
    # No, L-BFGS-B handles bounds. DE handles bounds.
    # Let's fix theta1 = 0 to reduce dimensionality and avoid redundant rotations.
    # If we fix theta1=0, we only need to search r1, r2, theta2.
    
    def _neg_log_score_3d(params_3d):
        r1, r2, theta2 = params_3d
        params_4d = [r1, r2, 0.0, theta2]
        return _neg_log_score_two_square(params_4d)

    def _points_from_3d(params_3d):
        r1, r2, theta2 = params_3d
        params_4d = [r1, r2, 0.0, theta2]
        return _two_square_points(params_4d)

    bounds_3d = [
        (0.1, 5.0),  # r1
        (0.1, 5.0),  # r2
        (0.0, np.pi/4) # theta2 relative to theta1=0. 
                       # Symmetry: rotating square 2 by pi/2 is same set. 
                       # Relative angle matters. 0 to pi/4 covers unique configurations?
                       # If theta2 = pi/4, it's regular octagon (if r1=r2).
                       # If theta2 = 0, it's two overlapping squares (4 distinct points? No, 8 points but pairs coincide if r1=r2? 
                       # If r1 != r2, they are distinct. 
                       # Let's allow 0 to 2*pi for safety, or 0 to pi/2?
                       # The set of points is invariant under rotation of both squares.
                       # We fixed theta1=0.
                       # The relative angle theta2 determines the shape.
                       # theta2 and theta2 + pi/2 give the same set of relative positions?
                       # Square 2 vertices: theta2, theta2+pi/2, ...
                       # If we add pi/2 to theta2, the set of angles is the same.
                       # So theta2 in [0, pi/2) is sufficient.
                       # Let's use [0, pi/2].
    ]
    bounds_3d[2] = (0.0, np.pi/2)

    # Phase 1: Differential Evolution on 3D space
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                _neg_log_score_3d,
                bounds=bounds_3d,
                seed=42,
                maxiter=300,
                tol=1e-10,
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < _neg_log_score_3d([1.0, 1.0, np.pi/8]): # Regular octagon score
                candidate_params = de_result.x
                candidate_pts = _points_from_3d(candidate_params)
                cand_score = _score(candidate_pts)
                if cand_score > best_score:
                    best_pts = candidate_pts
                    best_score = cand_score
    except Exception:
        pass

    # Phase 2: L-BFGS-B Local Refinement on 3D space
    if time.time() < deadline - 20:
        try:
            # Start from DE result if available, else regular octagon
            # We need to store the best params found
            # Re-run DE quickly or just use the result? 
            # The variable de_result is local to the try block. 
            # Let's structure it better.
            
            # I will re-organize the code to keep the best params.
            pass
        except Exception:
            pass
            
    # Let's rewrite construct to keep track of best params properly.
    
    # Reset best params to regular octagon equivalent in 3D space
    # Regular octagon: r1=1, r2=1, theta2=pi/8
    best_params_3d = np.array([1.0, 1.0, np.pi/8])
    best_neg_log = _neg_log_score_3d(best_params_3d)
    
    # Phase 1: DE
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                _neg_log_score_3d,
                bounds=bounds_3d,
                seed=42,
                maxiter=300,
                tol=1e-10,
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < best_neg_log:
                best_params_3d = de_result.x
                best_neg_log = de_result.fun
    except Exception:
        pass

    # Phase 2: L-BFGS-B
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _neg_log_score_3d,
                best_params_3d,
                method='L-BFGS-B',
                bounds=bounds_3d,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_neg_log:
                best_params_3d = res.x
                best_neg_log = res.fun
        except Exception:
            pass
            
    # Phase 3: High precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_log_score_3d,
                best_params_3d,
                method='L-BFGS-B',
                bounds=bounds_3d,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
            )
            if res.fun < best_neg_log:
                best_params_3d = res.x
                best_neg_log = res.fun
        except Exception:
            pass

    # Generate final points
    final_pts = _points_from_3d(best_params_3d)
    final_score = _score(final_pts)
    
    # Compare with baseline just in case
    if final_score > best_score:
        best_pts = final_pts
        best_score = final_score

    # Scale to diameter 2
    pts = np.array(best_pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(8, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-12:
        pts = pts * (2.0 / diameter)

    return [[float(p[0]), float(p[1])] for p in pts]
