"""Optimized 8-point configuration maximizing product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    timeout = 140.0  # 10s margin before 150s kill

    n = 8

    def exact_log_score(points):
        """Exact log score: log(prod_{i!=j} d'_ij / n^n) where d'_ij scaled to diam 2."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        if len(off_diag_dists) == 0:
            return -np.inf
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return -np.inf
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        log_prod = np.sum(np.log(scaled_dists))
        return log_prod - n * math.log(n)

    def soft_log_score(points, beta=50.0):
        """Soft-max diameter approximation for smooth gradient-based optimization."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        if len(off_diag_dists) == 0:
            return -np.inf
        scaled = off_diag_dists / beta
        m = np.max(scaled)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
        diameter = beta * log_sum_exp
        if diameter < 1e-10:
            return -np.inf
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        return np.sum(np.log(scaled_dists)) - n * math.log(n)

    def objective_soft(x, beta=50.0):
        points = x.reshape(n, 2)
        return -soft_log_score(points, beta)

    def objective_exact(x):
        points = x.reshape(n, 2)
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 1e10
        s = exact_log_score(points)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = exact_log_score(reg_oct)
    best_points = reg_oct.copy()

    def try_update(points):
        nonlocal best_score, best_points
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        if np.min(dists[mask]) < 1e-10:
            return
        s = exact_log_score(points)
        if np.isfinite(s) and s > best_score:
            best_score = s
            best_points = points.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Two squares with various relative rotations and radius ratios
    for rot in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for r2 in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
            sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
            sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
            starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal pairs: 4 pairs of antipodal points with varying angles
    for seed_val in range(8):
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Rectangular grid-like patterns
    for s in [0.5, 0.7, 1.0, 1.2, 1.5]:
        for yf in [0.5, 0.7, 0.9]:
            xs = np.array([-s, -s, s, s, -s*0.5, s*0.5, -s*0.5, s*0.5])
            ys = np.array([-s, s, -s, s, -s*yf, s*yf, -s*yf, s*yf])
            starts.append(np.column_stack([xs, ys]).flatten())

    # Random points
    np.random.seed(42)
    for _ in range(20):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())

    # Points on circle with perturbations
    for _ in range(15):
        base_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.15, 0.15, n)
        radii = 1.0 + np.random.uniform(-0.15, 0.15, n)
        pts = np.column_stack([np.cos(base_angles) * radii, np.sin(base_angles) * radii])
        starts.append(pts.flatten())

    # Phase 1: Local optimization with soft-max surrogate (L-BFGS-B)
    for s in starts:
        if time.time() - start_time > timeout - 40:
            break
        try:
            res = minimize(
                lambda x: objective_soft(x, beta=50.0),
                s,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < np.inf:
                try_update(res.x.reshape(n, 2))
        except Exception:
            pass

    # Phase 2: Dedicated two-square parametric optimization
    # Parameters: theta (rotation), r1 (radius sq1), r2 (radius sq2)
    if time.time() - start_time < timeout - 30:
        def two_square_score(params):
            theta, r1, r2 = params
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + theta)
            sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
            sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
            pts = np.vstack([sq1, sq2])
            return -exact_log_score(pts)

        # Grid search over theta and r2 (r1=1 by scale invariance)
        theta_grid = np.linspace(0, math.pi/4, 25)
        r2_grid = np.linspace(0.5, 1.5, 25)
        best_ts_score = -np.inf
        best_ts_params = None
        for th in theta_grid:
            for r2v in r2_grid:
                s = -two_square_score([th, 1.0, r2v])
                if s > best_ts_score:
                    best_ts_score = s
                    best_ts_params = [th, 1.0, r2v]

        if best_ts_params is not None:
            # Refine with L-BFGS-B
            try:
                ts_res = minimize(
                    two_square_score,
                    best_ts_params,
                    method='L-BFGS-B',
                    bounds=[(0, math.pi/4), (0.1, 5), (0.1, 5)],
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if ts_res.fun < np.inf:
                    theta, r1, r2 = ts_res.x
                    a1 = 2 * np.pi * np.arange(4) / 4
                    a2 = 2 * np.pi * (np.arange(4) / 4 + theta)
                    sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
                    sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
                    try_update(np.vstack([sq1, sq2]))
            except Exception:
                pass

    # Phase 3: Antipodal pair parametric optimization
    # 4 antipodal pairs at angles alpha_0..alpha_3 on unit circle
    if time.time() - start_time < timeout - 25:
        def antipodal_score(params):
            alphas = np.sort(params)
            all_angles = np.concatenate([alphas, alphas + np.pi])
            pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
            return -exact_log_score(pts)

        # Grid search
        best_ap_score = -np.inf
        best_ap_params = None
        for seed_val in range(10):
            np.random.seed(seed_val + 100)
            alphas = np.sort(np.random.uniform(0, np.pi, 4))
            s = -antipodal_score(alphas)
            if s > best_ap_score:
                best_ap_score = s
                best_ap_params = alphas.copy()

        if best_ap_params is not None:
            try:
                ap_res = minimize(
                    antipodal_score,
                    best_ap_params,
                    method='L-BFGS-B',
                    bounds=[(0, np.pi)] * 4,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if ap_res.fun < np.inf:
                    alphas = np.sort(ap_res.x)
                    all_angles = np.concatenate([alphas, alphas + np.pi])
                    try_update(np.column_stack([np.cos(all_angles), np.sin(all_angles)]))
            except Exception:
                pass

    # Phase 4: Refine with higher beta (closer to exact diameter)
    if time.time() - start_time < timeout - 20:
        for beta_val in [200.0, 500.0, 1000.0, 5000.0]:
            try:
                res = minimize(
                    lambda x: objective_soft(x, beta=beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # Phase 5: Global optimization with differential evolution
    if time.time() - start_time < timeout - 30:
        bounds = [(-5, 5)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: objective_soft(x, beta=100.0),
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True
            )
            if de_res.fun < np.inf:
                try_update(de_res.x.reshape(n, 2))
        except Exception:
            pass

        # Second DE run with different seed
        if time.time() - start_time < timeout - 15:
            try:
                de_res2 = differential_evolution(
                    lambda x: objective_soft(x, beta=100.0),
                    bounds,
                    seed=123,
                    maxiter=150,
                    popsize=12,
                    tol=1e-10,
                    polish=True
                )
                if de_res2.fun < np.inf:
                    try_update(de_res2.x.reshape(n, 2))
            except Exception:
                pass

    # Phase 6: High-precision polish with exact diameter
    if time.time() - start_time < timeout - 5:
        for method in ['L-BFGS-B', 'BFGS', 'CG']:
            try:
                final_res = minimize(
                    objective_exact,
                    best_points.flatten(),
                    method=method,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if final_res.fun < np.inf:
                    try_update(final_res.x.reshape(n, 2))
            except Exception:
                pass

    # Final validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    # Scale/Check bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
