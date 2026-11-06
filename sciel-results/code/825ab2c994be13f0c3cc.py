import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts, dtype=float)
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
        
        # Score = (Prod_{i!=j} d'_ij) / n^n
        # d'_ij = 2 * d_ij / D
        # Prod_{i!=j} d'_ij = (2/D)^{n(n-1)} * Prod_{i!=j} d_ij
        # Prod_{i!=j} d_ij = (Prod_{i<j} d_ij)^2
        # log(Score) = n(n-1)*(log(2) - log(D)) + 2*sum(log(d_ij)) - n*log(n)
        
        log_prod_unordered = np.sum(np.log(d))
        log_prod_ordered = 2.0 * log_prod_unordered
        log_scale_factor = n * (n - 1) * (np.log(2.0) - np.log(D))
        log_norm = -n * np.log(n)
        
        log_score = log_prod_ordered + log_scale_factor + log_norm
        return math.exp(log_score)

    def neg_log_score_full(x):
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
        
        log_prod_unordered = np.sum(np.log(d))
        log_prod_ordered = 2.0 * log_prod_unordered
        log_scale_factor = n * (n - 1) * (np.log(2.0) - np.log(D))
        log_norm = -n * np.log(n)
        log_score = log_prod_ordered + log_scale_factor + log_norm
        return -log_score

    def neg_log_score_two_squares(params):
        """
        Objective for two-squares model.
        params: [r1, r2, phi]
        """
        r1, r2, phi = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
            
        # Generate points
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + phi
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        
        # Compute score
        s = compute_score(pts)
        if s <= 0:
            return 1e10
        return -math.log(s)

    def neg_log_score_symmetric(x):
        """
        Objective for reflection-symmetric model.
        x = [x1, y1, x2, y2, x3, y3, x4, y4]
        Points are (xk, yk) and (xk, -yk)
        """
        pts = np.array([
            [x[0], x[1]],
            [x[0], -x[1]],
            [x[2], x[3]],
            [x[2], -x[3]],
            [x[4], x[5]],
            [x[4], -x[5]],
            [x[6], x[7]],
            [x[6], -x[7]]
        ])
        return neg_log_score_full(pts.flatten())

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

    # 2. Two Squares Optimization with Differential Evolution
    # Search r1, r2 in [0.1, 5.0], phi in [0, pi/2]
    if time.time() < deadline - 60.0:
        try:
            bounds_ts = [(0.1, 5.0), (0.1, 5.0), (0.0, np.pi/2)]
            res_de = differential_evolution(
                neg_log_score_two_squares,
                bounds_ts,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=True
            )
            if np.all(np.isfinite(res_de.x)):
                r1, r2, phi = res_de.x
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + phi
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts_sym = np.vstack([pts1, pts2])
                update_best(pts_sym)
                
                # Refine with Nelder-Mead
                if time.time() < deadline - 40.0:
                    res_nm = minimize(
                        neg_log_score_two_squares,
                        res_de.x,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if np.all(np.isfinite(res_nm.x)):
                        r1, r2, phi = res_nm.x
                        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                        angles2 = angles1 + phi
                        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                        pts_sym = np.vstack([pts1, pts2])
                        update_best(pts_sym)
        except Exception:
            pass

    # 3. Reflection-Symmetric Optimization with Differential Evolution
    if time.time() < deadline - 50.0:
        try:
            # Initial bounds for symmetric points
            bounds_sym = [(-5, 5)] * 8
            res_de_sym = differential_evolution(
                neg_log_score_symmetric,
                bounds_sym,
                seed=123,
                maxiter=100,
                tol=1e-8,
                polish=True
            )
            if np.all(np.isfinite(res_de_sym.x)):
                x_opt = res_de_sym.x
                pts = np.array([
                    [x_opt[0], x_opt[1]],
                    [x_opt[0], -x_opt[1]],
                    [x_opt[2], x_opt[3]],
                    [x_opt[2], -x_opt[3]],
                    [x_opt[4], x_opt[5]],
                    [x_opt[4], -x_opt[5]],
                    [x_opt[6], x_opt[7]],
                    [x_opt[6], -x_opt[7]]
                ])
                update_best(pts)
                
                # Refine with Nelder-Mead
                if time.time() < deadline - 30.0:
                    res_nm_sym = minimize(
                        neg_log_score_symmetric,
                        x_opt,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if np.all(np.isfinite(res_nm_sym.x)):
                        x_opt2 = res_nm_sym.x
                        pts2 = np.array([
                            [x_opt2[0], x_opt2[1]],
                            [x_opt2[0], -x_opt2[1]],
                            [x_opt2[2], x_opt2[3]],
                            [x_opt2[2], -x_opt2[3]],
                            [x_opt2[4], x_opt2[5]],
                            [x_opt2[4], -x_opt2[5]],
                            [x_opt2[6], x_opt2[7]],
                            [x_opt2[6], -x_opt2[7]]
                        ])
                        update_best(pts2)
        except Exception:
            pass

    # 4. Full Coordinate Optimization from Best Point
    if best_pts is not None and time.time() < deadline - 20.0:
        x0 = best_pts.flatten()
        try:
            # L-BFGS-B
            res = minimize(
                neg_log_score_full,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                update_best(refined_pts)
        except Exception:
            pass
        
        # Try Nelder-Mead on full coordinates from best point
        if time.time() < deadline - 10.0:
            try:
                res_nm = minimize(
                    neg_log_score_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if np.all(np.isfinite(res_nm.x)):
                    refined_pts_nm = res_nm.x.reshape(n, 2)
                    update_best(refined_pts_nm)
            except Exception:
                pass

    # 5. Random Restarts in Full Space
    if time.time() < deadline - 5.0:
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            # Random points in unit square, scaled
            pts_rand = np.random.rand(n, 2) * 2 - 1
            x0 = pts_rand.flatten()
            try:
                res = minimize(
                    neg_log_score_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    update_best(refined_pts)
            except Exception:
                pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Ensure distinct points
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :]
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
