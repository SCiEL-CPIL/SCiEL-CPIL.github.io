import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    timeout = 140.0  # 10s margin before 150s kill

    n = 8

    def log_product(points, beta=100.0):
        """
        Compute log product of pairwise distances after scaling to diameter 2.
        Uses smooth soft-max for diameter when beta is finite,
        or exact max when beta is large.
        """
        # points: (n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)

        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]

        if len(off_diag_dists) == 0:
            return -np.inf

        # Compute diameter
        if beta < 1e6:
            # Smooth soft-max approximation of max
            scaled = off_diag_dists / beta
            m = np.max(scaled)
            log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
            diameter = beta * log_sum_exp
        else:
            diameter = np.max(off_diag_dists)

        if diameter < 1e-10:
            return -np.inf

        # Scale to diameter 2
        scaled_dists = off_diag_dists * (2.0 / diameter)

        # Avoid log(0)
        scaled_dists = np.maximum(scaled_dists, 1e-15)

        return np.sum(np.log(scaled_dists))

    def objective(x, beta=100.0):
        points = x.reshape(n, 2)
        lp = log_product(points, beta)
        return -lp

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = log_product(reg_oct, beta=1e8)
    best_points = reg_oct.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Two squares with various relative rotations
    for rot in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal pairs: 4 pairs of antipodal points with varying angles
    for seed_val in [0, 1, 2]:
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Rectangular grid-like patterns
    for s in [0.5, 1.0, 1.5]:
        xs = np.array([-s, -s, s, s, -s*0.5, s*0.5, -s*0.5, s*0.5])
        ys = np.array([-s, s, -s, s, -s*0.7, s*0.7, -s*0.7, s*0.7])
        starts.append(np.column_stack([xs, ys]).flatten())

    # Random points
    np.random.seed(42)
    for _ in range(15):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())

    # Points on circle with perturbations
    for _ in range(10):
        base_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.1, 0.1, n)
        radii = 1.0 + np.random.uniform(-0.1, 0.1, n)
        pts = np.column_stack([np.cos(base_angles) * radii, np.sin(base_angles) * radii])
        starts.append(pts.flatten())

    # Phase 1: Local optimization with soft-max surrogate (L-BFGS-B)
    # This enables gradient-based optimization that can escape local minima
    for s in starts:
        if time.time() - start_time > timeout - 30:
            break
        try:
            # Use soft-max with moderate beta for smooth optimization
            res = minimize(
                lambda x: objective(x, beta=50.0),
                s,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < np.inf:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 2: Refine with higher beta (closer to exact diameter)
    if time.time() - start_time < timeout - 20:
        for beta_val in [200.0, 500.0, 1000.0]:
            try:
                res = minimize(
                    lambda x: objective(x, beta=beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 3: Global optimization with differential evolution
    if time.time() - start_time < timeout - 25:
        bounds = [(-5, 5)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: objective(x, beta=100.0),
                bounds,
                seed=42,
                maxiter=300,
                popsize=15,
                tol=1e-10,
                polish=True
            )
            if de_res.fun < np.inf:
                score = -de_res.fun
                if score > best_score:
                    best_score = score
                    best_points = de_res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 4: High-precision polish with exact diameter
    if time.time() - start_time < timeout - 5:
        for method in ['L-BFGS-B', 'BFGS']:
            try:
                final_res = minimize(
                    lambda x: objective(x, beta=1e8),
                    best_points.flatten(),
                    method=method,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if final_res.fun < np.inf:
                    score = -final_res.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_res.x.reshape(n, 2)
            except Exception:
                pass

    # Validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    # Scale/Check bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
