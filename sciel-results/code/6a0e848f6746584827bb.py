import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    n = 8

    def compute_score(pts):
        """Calculate the actual score: Delta / n^n where Delta is product of pairwise distances after scaling to diameter 2."""
        pts = np.array(pts, dtype=np.float64)
        # Compute pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        D = np.max(d)
        if D < 1e-9:
            return -np.inf
        # Scale so diameter = 2
        scale = 2.0 / D
        d_scaled = d * scale
        log_delta = np.sum(np.log(d_scaled))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        return score

    def two_squares(r1, r2, theta1, theta2):
        """Generate 8 points as two concentric squares with independent radii and rotations."""
        # Square 1: 4 points at radius r1, rotation theta1
        angles1 = theta1 + np.pi / 2 * np.arange(4)
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        # Square 2: 4 points at radius r2, rotation theta2
        angles2 = theta2 + np.pi / 2 * np.arange(4)
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def neg_log_score(params):
        """Negative of log(Delta) - 28*log(D) for minimization. Returns large penalty if invalid."""
        if time.time() > deadline:
            return 1e10
        r1, r2, theta1, theta2 = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
        pts = two_squares(r1, r2, theta1, theta2)
        # Check distinctness
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 1e10
        D = np.max(d)
        if D < 1e-9:
            return 1e10
        log_prod = np.sum(np.log(d))
        # We want to maximize log_prod - 28*log(D), so minimize 28*log(D) - log_prod
        return 28.0 * np.log(D) - log_prod

    # Phase 1: Exhaustive grid search over the 4D parameter space
    # By symmetry, we can assume r1 >= r2 >= 0, and theta1, theta2 in [0, pi/2)
    # due to the 4-fold symmetry of squares.
    # Actually, let's be more careful. The two squares can have any relative rotation.
    # Let's search r1 in [0.5, 2], r2 in [0.1, r1], theta1 in [0, pi/2], theta2 in [0, pi/2]
    
    best_score = -np.inf
    best_params = None
    
    # Coarse grid first
    n_r1 = 20
    n_r2 = 20
    n_theta = 18
    
    r1_vals = np.linspace(0.5, 2.0, n_r1)
    r2_vals = np.linspace(0.1, 1.5, n_r2)
    theta_vals = np.linspace(0.0, np.pi/2, n_theta)
    
    for r1 in r1_vals:
        if time.time() > deadline - 30:
            break
        for r2 in r2_vals:
            if r2 > r1:
                continue
            for t1 in theta_vals:
                if time.time() > deadline - 25:
                    break
                for t2 in theta_vals:
                    params = np.array([r1, r2, t1, t2])
                    obj = neg_log_score(params)
                    if obj < 1e9:
                        pts = two_squares(r1, r2, t1, t2)
                        sc = compute_score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_params = params.copy()
    
    # Also try the regular octagon as baseline
    angles_oct = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    oct_score = compute_score(octagon)
    if oct_score > best_score:
        best_score = oct_score
        best_pts = octagon
    else:
        best_pts = two_squares(best_params[0], best_params[1], best_params[2], best_params[3])
    
    # Phase 2: Local refinement from the best grid point
    if best_params is not None and time.time() < deadline - 15:
        # Refine in the 4D parameter space
        bounds = [(0.1, 3.0), (0.05, 3.0), (0.0, np.pi), (0.0, np.pi)]
        
        try:
            res = minimize(
                neg_log_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < 1e9:
                cand_pts = two_squares(res.x[0], res.x[1], res.x[2], res.x[3])
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
        
        # Also try L-BFGS-B for faster convergence
        try:
            res2 = minimize(
                neg_log_score,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.fun < 1e9:
                cand_pts = two_squares(res2.x[0], res2.x[1], res2.x[2], res2.x[3])
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
    
    # Phase 3: Full 16D local refinement from the best point found
    if time.time() < deadline - 10:
        def full_neg_log_score(pts_flat):
            if time.time() > deadline:
                return 1e10
            pts = pts_flat.reshape(n, 2)
            dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            iu = np.triu_indices(n, 1)
            d = dists[iu]
            if np.min(d) < 1e-8:
                return 1e10
            D = np.max(d)
            if D < 1e-9:
                return 1e10
            log_prod = np.sum(np.log(d))
            return 28.0 * np.log(D) - log_prod
        
        best_flat = best_pts.flatten()
        
        try:
            res3 = minimize(
                full_neg_log_score,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res3.fun < 1e9:
                cand_pts = res3.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
        
        try:
            res4 = minimize(
                full_neg_log_score,
                best_flat,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res4.fun < 1e9:
                cand_pts = res4.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
    
    # Phase 4: Differential evolution in 4D parameter space if time permits
    if time.time() < deadline - 20:
        bounds_4d = [(0.1, 3.0), (0.05, 3.0), (0.0, np.pi), (0.0, np.pi)]
        try:
            res_de = differential_evolution(
                neg_log_score,
                bounds_4d,
                seed=42,
                maxiter=100,
                tol=1e-10,
                popsize=10,
                polish=True,
                polish_options={'method': 'Nelder-Mead', 'options': {'maxiter': 2000}}
            )
            if res_de.fun < 1e9:
                cand_pts = two_squares(res_de.x[0], res_de.x[1], res_de.x[2], res_de.x[3])
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
    
    # Phase 5: Final full 16D polish from the absolute best
    if time.time() < deadline - 5:
        best_flat = best_pts.flatten()
        try:
            res_final = minimize(
                full_neg_log_score,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res_final.fun < 1e9:
                cand_pts = res_final.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts.tolist()
