import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # 1. Search in the 3D space of two concentric squares
    # Parameters: R1, R2, phi
    # Square 1 angles: 0, pi/2, pi, 3pi/2
    # Square 2 angles: phi, phi+pi/2, phi+pi, phi+3pi/2
    
    def score_3d(params):
        R1, R2, phi = params
        if R1 < 0 or R2 < 0:
            return 0.0 # Penalty for invalid radii
        
        # Generate points
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = phi + angles1
        
        pts1 = np.column_stack([R1 * np.cos(angles1), R1 * np.sin(angles1)])
        pts2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
        
        pts = np.vstack([pts1, pts2])
        
        # Calculate diameter
        # Pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        D = np.max(dists)
        
        if D == 0:
            return 0.0
            
        # Log product of distances
        # We only need i < j
        triu_indices = np.triu_indices(8, k=1)
        prod_dists = np.prod(dists[triu_indices])
        
        if prod_dists <= 0:
            return 0.0
            
        log_score = np.log(prod_dists) - 28 * np.log(D)
        
        return log_score

    # Bounds for R1, R2, phi
    # R1, R2 should be positive. Let's say 0.1 to 5.0
    # phi is in [0, pi/2] due to symmetry (square symmetry is pi/2)
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, np.pi/2)]
    
    best_result_3d = None
    best_score_3d = -np.inf
    
    if time.time() < deadline - 20:
        try:
            res = differential_evolution(
                lambda x: -score_3d(x), 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15,
                tol=1e-8,
                polish=True
            )
            best_score_3d = -res.fun
            best_params_3d = res.x
            best_result_3d = best_params_3d
        except Exception as e:
            pass

    # Initialize points from best 3D result
    if best_result_3d is not None:
        R1, R2, phi = best_result_3d
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = phi + angles1
        pts1 = np.column_stack([R1 * np.cos(angles1), R1 * np.sin(angles1)])
        pts2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
    else:
        # Fallback: Regular Octagon
        k = np.arange(8)
        angles = 2 * np.pi * k / 8
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Scale to diameter 2
    dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
    D = np.max(dists)
    pts = pts * (2.0 / D)
    
    # 2. Local Refinement in 16D
    # Maximize log product of distances, subject to diameter <= 2 (or normalize)
    # It is better to optimize the normalized score directly or use a penalty.
    # Let's optimize the log score: sum(log(d_ij)) - 28*log(D)
    
    def score_16d(flat_pts):
        if time.time() > deadline - 5:
            return 0.0
        p = flat_pts.reshape(8, 2)
        dists = np.linalg.norm(p[:, np.newaxis] - p[np.newaxis, :], axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        triu_indices = np.triu_indices(8, k=1)
        prod_dists = np.prod(dists[triu_indices])
        if prod_dists <= 0:
            return 0.0
        return np.log(prod_dists) - 28 * np.log(D)

    best_flat = pts.flatten()
    best_score_16d = score_16d(best_flat)
    
    # Try a few local optimizations with different methods
    methods = ['L-BFGS-B', 'Nelder-Mead']
    for method in methods:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                lambda x: -score_16d(x), 
                best_flat, 
                method=method, 
                options={'maxiter': 500, 'disp': False}
            )
            if res.fun < -best_score_16d: # i.e., score improved
                best_score_16d = -res.fun
                best_flat = res.x
        except Exception:
            pass

    final_pts = best_flat.reshape(8, 2)
    
    # Final scaling to ensure diameter is exactly 2
    dists = np.linalg.norm(final_pts[:, np.newaxis] - final_pts[np.newaxis, :], axis=2)
    D = np.max(dists)
    if D > 0:
        final_pts = final_pts * (2.0 / D)
        
    # Convert to list of lists of floats
    return [[float(x), float(y)] for x, y in final_pts]
