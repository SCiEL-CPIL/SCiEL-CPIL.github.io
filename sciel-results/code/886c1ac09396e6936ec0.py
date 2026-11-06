import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_score(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2
        return Delta / (n ** n)

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_candidate(reg_pts)

    # 2. Two Concentric Squares with refined search
    def objective_two_squares(params):
        r1, r2, theta = params
        if r1 < 0.001 or r2 < 0.001:
            return 1e10
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Extended guesses
    guesses = [
        [0.75, 1.05, np.pi/4],
        [0.85, 0.95, np.pi/4],
        [0.95, 0.85, np.pi/4],
        [0.70, 1.10, np.pi/5],
        [0.80, 1.00, np.pi/6],
        [1.00, 0.80, np.pi/6],
        [0.65, 1.15, np.pi/7],
        [0.90, 0.90, np.pi/4],
        [0.75, 1.00, np.pi/3],
        [1.00, 0.75, np.pi/3],
        [0.82, 1.02, np.pi/4],
        [0.88, 0.92, np.pi/4],
        [0.78, 1.08, 3*np.pi/8],
        [0.92, 0.88, np.pi/5],
        [0.85, 0.95, 3*np.pi/8],
        [0.72, 1.12, np.pi/7],
        [0.80, 1.00, 5*np.pi/16],
        [1.00, 0.80, 3*np.pi/10],
    ]

    for guess in guesses:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                objective_two_squares,
                guess,
                method='L-BFGS-B',
                bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/2)],
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.success or res.fun < 1e9:
                r1, r2, theta = res.x
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + theta
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts = np.vstack([pts1, pts2])
                try_candidate(pts)
        except Exception:
            pass

    # 3. Conjugate-symmetric 8D polynomial representation
    def poly_objective(params):
        if not np.all(np.isfinite(params)):
            return 1e10
        complex_roots = []
        for i in range(4):
            re = params[2*i]
            im = params[2*i+1]
            z = complex(re, im)
            complex_roots.append(z)
            complex_roots.append(z.conjugate())
        pts = np.array([[z.real, z.imag] for z in complex_roots])
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
        log_sum = np.sum(np.log(d))
        log_score = 2.0 * (log_sum + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        coord_penalty = 0.0
        if np.max(np.abs(pts)) > 5.0:
            coord_penalty = 10.0 * (np.max(np.abs(pts)) - 5.0)**2
        return -log_score + coord_penalty

    # Poly initializations
    c = np.cos(np.pi/4)
    s = np.sin(np.pi/4)
    poly_inits = [
        [1.0, 0.0, c, s, 0.0, 1.0, -c, s],
        [1.0, 0.1, c+0.1, s-0.1, 0.1, 1.0, -c-0.1, s+0.1],
        [1.0, 0.0, 0.7, 0.7, 0.0, 0.7, -0.7, 0.7],
        [1.0, 0.0, 0.8, 0.6, 0.0, 0.8, -0.6, 0.8],
        [0.9, 0.1, c*0.9, s*0.9, 0.0, 0.95, -c*0.9, s*0.9],
        [1.0, 0.05, c*1.05, s*0.95, 0.05, 1.0, -c*1.05, s*1.05],
    ]

    np.random.seed(123)
    for _ in range(8):
        re_vals = np.random.uniform(-1.5, 1.5, 4)
        im_vals = np.random.uniform(-1.5, 1.5, 4)
        params = [re_vals[0], im_vals[0], re_vals[1], im_vals[1], re_vals[2], im_vals[2], re_vals[3], im_vals[3]]
        poly_inits.append(params)

    bounds_poly = [(-2.0, 2.0)] * 8

    for init in poly_inits:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                poly_objective,
                init,
                method='L-BFGS-B',
                bounds=bounds_poly,
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                complex_roots = []
                for i in range(4):
                    re = res.x[2*i]
                    im = res.x[2*i+1]
                    z = complex(re, im)
                    complex_roots.append(z)
                    complex_roots.append(z.conjugate())
                pts = np.array([[z.real, z.imag] for z in complex_roots])
                try_candidate(pts)
        except Exception:
            pass

    # 4. General 16D optimization with smooth diameter constraint
    def objective_general(x):
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
        log_sum = np.sum(np.log(d))
        # Smooth penalty for D > 2 using log-sum-exp
        d_sq = d**2
        max_d_sq = np.max(d_sq)
        # Soft penalty: smooth approximation of (D-2)^2 when D>2
        penalty = 0.0
        if D > 2.0:
            penalty = 50.0 * (D - 2.0)**2
        return -log_sum + penalty

    def objective_general_smooth(x):
        """Version with fully smooth log-sum-exp diameter penalty."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diffs**2, axis=2)
        iu = np.triu_indices(n, k=1)
        d_sq_u = d_sq[iu]
        if np.min(d_sq_u) < 1e-10:
            return 1e10
        # Smooth max via log-sum-exp
        m = np.max(d_sq_u)
        if m < 1e-10:
            return 1e10
        log_sum_exp = m + np.log(np.sum(np.exp(d_sq_u - m)))
        D_smooth = np.sqrt(log_sum_exp)
        d_u = np.sqrt(d_sq_u)
        log_sum = np.sum(np.log(d_u))
        penalty = 0.0
        if D_smooth > 2.0:
            penalty = 50.0 * (D_smooth - 2.0)**2
        return -log_sum + penalty

    inits = []
    if best_pts is not None:
        inits.append(best_pts.copy())
        np.random.seed(42)
        for scale in [0.05, 0.1, 0.15, 0.2]:
            inits.append(best_pts + scale * np.random.randn(n, 2))

    # Random inits
    np.random.seed(7)
    for _ in range(8):
        r = np.random.rand(n) * 1.2
        theta = np.random.rand(n) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)

    # Perturbed regular octagon
    for _ in range(4):
        angles = angles_reg + 0.15 * np.random.randn(n)
        radii = 1.0 + 0.15 * np.random.randn(n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    bounds_general = [(-1.5, 1.5)] * (2 * n)

    for init in inits:
        if time.time() > deadline - 15.0:
            break
        try:
            x0 = init.flatten()
            res = minimize(
                objective_general,
                x0,
                method='L-BFGS-B',
                bounds=bounds_general,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

    # 5. Fine grid search in two-squares family
    if time.time() < deadline - 25.0:
        r1_vals = np.linspace(0.70, 1.10, 12)
        r2_vals = np.linspace(0.70, 1.10, 12)
        theta_vals = np.linspace(0.2, np.pi/2 - 0.2, 8)
        for r1 in r1_vals:
            for r2 in r2_vals:
                for theta in theta_vals:
                    if time.time() > deadline - 15.0:
                        break
                    guess = [r1, r2, theta]
                    try:
                        res = minimize(
                            objective_two_squares,
                            guess,
                            method='L-BFGS-B',
                            bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/2)],
                            options={'maxiter': 300, 'ftol': 1e-15}
                        )
                        if res.success or res.fun < 1e9:
                            r1_opt, r2_opt, theta_opt = res.x
                            angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                            angles2 = angles1 + theta_opt
                            pts1 = np.column_stack([r1_opt * np.cos(angles1), r1_opt * np.sin(angles1)])
                            pts2 = np.column_stack([r2_opt * np.cos(angles2), r2_opt * np.sin(angles2)])
                            pts = np.vstack([pts1, pts2])
                            try_candidate(pts)
                    except Exception:
                        pass

    # 6. Final high-precision refinement passes
    if best_pts is not None:
        # Pass 1: Tight L-BFGS-B
        try:
            res = minimize(
                objective_general,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_general,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

        # Pass 2: Nelder-Mead polish with smooth objective
        if time.time() < deadline - 10.0:
            try:
                res = minimize(
                    objective_general_smooth,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'fatol': 1e-15, 'xatol': 1e-15, 'adaptive': True}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    try_candidate(refined_pts)
            except Exception:
                pass

        # Pass 3: Another L-BFGS-B from polished result
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    objective_general,
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds_general,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    try_candidate(refined_pts)
            except Exception:
                pass

        # Pass 4: Tiny perturbations and re-optimize
        if time.time() < deadline - 5.0:
            np.random.seed(99)
            for perturb_scale in [0.001, 0.005, 0.01]:
                if time.time() > deadline - 3.0:
                    break
                try:
                    perturbed = best_pts + perturb_scale * np.random.randn(n, 2)
                    res = minimize(
                        objective_general,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds_general,
                        options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
                    )
                    if np.all(np.isfinite(res.x)):
                        refined_pts = res.x.reshape(n, 2)
                        try_candidate(refined_pts)
                except Exception:
                    pass

        # Pass 5: Final Nelder-Mead with very tight tolerance
        if time.time() < deadline - 3.0:
            try:
                res = minimize(
                    objective_general_smooth,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'fatol': 1e-16, 'xatol': 1e-16, 'adaptive': True}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    try_candidate(refined_pts)
            except Exception:
                pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
