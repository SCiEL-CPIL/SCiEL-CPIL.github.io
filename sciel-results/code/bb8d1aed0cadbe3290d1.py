import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Check distinctness
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-12:
            return 0.0
        
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        
        # Scale to diameter 2
        # Score = (Prod_{i!=j} d'_ij) / n^n
        # d'_ij = 2 * d_ij / D
        # Prod_{i!=j} d'_ij = (2/D)^{n(n-1)} * Prod_{i!=j} d_ij
        # Prod_{i!=j} d_ij = (Prod_{i<j} d_ij)^2
        # log(Score) = n(n-1)*(log(2) - log(D)) + 2*sum(log(d_ij)) - n*log(n)
        
        # We compute log_score directly for stability
        log_prod_unordered = np.sum(np.log(d))
        log_prod_ordered = 2.0 * log_prod_unordered
        log_scale_factor = n * (n - 1) * (np.log(2.0) - np.log(D))
        log_norm = -n * np.log(n)
        
        log_score = log_prod_ordered + log_scale_factor + log_norm
        return math.exp(log_score)

    def objective_full(x):
        """Objective for full coordinate optimization: minimize negative log score."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        
        # Maximize: 2 * sum(log(d)) - n(n-1)*log(D)
        # Minimize: - [ 2 * sum(log(d)) - n(n-1)*log(D) ]
        val = 2.0 * np.sum(np.log(d)) - n * (n - 1) * np.log(D)
        return -val

    def objective_params(params):
        """Objective for parameterized optimization (two squares)."""
        r1, r2, theta = params
        if r1 < 0 or r2 < 0:
            return 1e10
        # Generate points
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        
        # Reuse logic from objective_full but on generated pts
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        val = 2.0 * np.sum(np.log(d)) - n * (n - 1) * np.log(D)
        return -val

    best_score = 0.0
    best_pts = None

    def update_best(pts):
        nonlocal best_score, best_pts
        if pts is None or not np.all(np.isfinite(pts)):
            return
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    update_best(reg_pts)

    # 2. Parameterized Search: Two Squares
    # Initial guesses for r1, r2, theta
    # r1, r2 in [0.5, 1.5], theta in [0, pi/2]
    # We use a grid search to find a good starting point for local optimization
    
    best_param_score = -np.inf
    best_params = None
    
    # Coarse grid
    r1_vals = np.linspace(0.5, 1.5, 10)
    r2_vals = np.linspace(0.5, 1.5, 10)
    theta_vals = np.linspace(0, np.pi/2, 10)
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for theta in theta_vals:
                if time.time() > deadline - 60.0:
                    break
                params = [r1, r2, theta]
                try:
                    obj_val = objective_params(params)
                    if obj_val < best_param_score:
                        best_param_score = obj_val
                        best_params = params
                except Exception:
                    pass
    
    if best_params is not None:
        # Refine parameters using Nelder-Mead
        try:
            res_param = minimize(
                objective_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_param.success and np.all(np.isfinite(res_param.x)):
                best_params = res_param.x
                # Generate points from best params
                r1, r2, theta = best_params
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + theta
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts_sym = np.vstack([pts1, pts2])
                update_best(pts_sym)
        except Exception:
            pass

    # 3. Full Coordinate Optimization
    # Start from best symmetric solution and regular octagon
    initial_guesses = []
    if best_pts is not None:
        initial_guesses.append(best_pts)
    initial_guesses.append(reg_pts)
    
    # Add a few random perturbations of the regular octagon
    np.random.seed(42)
    for _ in range(3):
        if time.time() > deadline - 30.0:
            break
        angles = angles_reg + 0.2 * np.random.randn(n)
        radii = 1.0 + 0.2 * np.random.randn(n)
        radii = np.abs(radii) + 0.1
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_guesses.append(pts)

    bounds = [(-5, 5)] * 16

    for init_pts in initial_guesses:
        if time.time() > deadline - 10.0:
            break
        
        x0 = init_pts.flatten()
        
        try:
            # Use L-BFGS-B for refinement
            res = minimize(
                objective_full,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                update_best(refined_pts)
                
                # If improved significantly, try a tighter refinement
                if best_score > 1.2:
                    if time.time() < deadline - 5.0:
                        res2 = minimize(
                            objective_full,
                            res.x,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
                        )
                        if np.all(np.isfinite(res2.x)):
                            refined_pts2 = res2.x.reshape(n, 2)
                            update_best(refined_pts2)
        except Exception:
            pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Ensure distinct points
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6:
        best_pts = reg_pts

    # Scale to ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
