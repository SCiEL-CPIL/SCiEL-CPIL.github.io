import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    n = 8
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s kill

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def score(points):
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    # Symmetric parametrization: p0=(-1,0), p1=(1,0),
    # 3 upper points (x_i, y_i) with y_i > 0, mirrored below.
    # x = [x1,y1,x2,y2,x3,y3]
    def build_points(x):
        pts = np.zeros((8, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [x[0], x[1]]
        pts[3] = [x[0], -x[1]]
        pts[4] = [x[2], x[3]]
        pts[5] = [x[2], -x[3]]
        pts[6] = [x[4], x[5]]
        pts[7] = [x[4], -x[5]]
        return pts

    def get_dists(x):
        pts = build_points(x)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return dists, mask

    # Smooth objective: soft-max diameter surrogate + log product
    def smooth_obj(x, beta=50.0):
        dists, mask = get_dists(x)
        d = dists[mask]
        if np.min(d) < 1e-10:
            return 1e10
        log_sum = np.sum(np.log(d))
        # soft-max: log-sum-exp of distances / beta
        dmax = np.max(d)
        soft_max = dmax + np.log(np.sum(np.exp((d - dmax) / beta))) / beta
        # We want to maximize log_sum - log(D) = log_sum - log(soft_max)
        # Minimize the negative
        return -(log_sum - np.log(soft_max))

    # Exact objective with penalty for Nelder-Mead
    def exact_obj(x):
        dists, mask = get_dists(x)
        d = dists[mask]
        D = np.max(d)
        if D == 0:
            return 1e10
        if np.min(d) < 1e-8:
            return 1e10
        log_sum = np.sum(np.log(d))
        return -(log_sum - np.log(D))

    def nm_penalized_obj(x):
        dists, mask = get_dists(x)
        d = dists[mask]
        D = np.max(d)
        if D == 0:
            return 1e10
        # Penalty for diameter > 2 (shouldn't happen with bounds, but safety)
        penalty = 0.0
        if D > 2.0 + 1e-6:
            penalty = 1e6 * (D - 2.0) ** 2
        min_d = np.min(d)
        if min_d < 1e-5:
            penalty += 1e6 * (1e-5 - min_d) ** 2
        log_sum = np.sum(np.log(np.maximum(d, 1e-12)))
        return -(log_sum - np.log(D)) + penalty

    bounds = [(-1.0, 1.0), (1e-3, 1.0),
              (-1.0, 1.0), (1e-3, 1.0),
              (-1.0, 1.0), (1e-3, 1.0)]

    # Regular octagon initial guess
    oct_pts = regular_octagon()
    # Octagon: (1,0),(.707,.707),(0,1),(-.707,.707),(-1,0),(-.707,-.707),(0,-1),(.707,-.707)
    # Upper points (excluding (1,0) and (-1,0)): (.707,.707), (0,1), (-.707,.707)
    init_oct = np.array([0.70710678, 0.70710678, 0.0, 1.0, -0.70710678, 0.70710678])

    best_score = score(regular_octagon())
    best_x = init_oct.copy()
    best_pts = build_points(best_x)

    def try_improve(x0, method, maxiter, ftol):
        nonlocal best_score, best_x, best_pts
        if time.time() >= deadline - 10:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(smooth_obj, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': ftol, 'gtol': ftol})
            elif method == 'Nelder-Mead':
                res = minimize(nm_penalized_obj, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': ftol,
                                        'fatol': ftol, 'adaptive': True})
            else:
                return
            if time.time() >= deadline - 5:
                return
            pts = build_points(res.x)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            D = np.max(dists)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if D <= 2.0 + 1e-4 and np.min(dists[mask]) > 1e-5:
                sc = score(pts)
                if sc > best_score:
                    best_score = sc
                    best_x = res.x.copy()
                    best_pts = pts
        except Exception:
            pass

    # Phase 1: L-BFGS-B from octagon init
    try_improve(init_oct, 'L-BFGS-B', 1000, 1e-14)

    # Phase 2: Nelder-Mead polish from best
    try_improve(best_x, 'Nelder-Mead', 5000, 1e-12)

    # Phase 3: Random restarts
    if time.time() < deadline - 30:
        np.random.seed(42)
        max_restarts = 25
        for k in range(max_restarts):
            if time.time() >= deadline - 25:
                break
            x0 = np.random.uniform(-0.9, 0.9, 3)
            y0 = np.random.uniform(0.15, 0.95, 3)
            x0_full = np.empty(6)
            x0_full[0::2] = x0
            x0_full[1::2] = y0
            # L-BFGS-B
            try_improve(x0_full, 'L-BFGS-B', 500, 1e-12)
            # NM polish
            if time.time() < deadline - 15:
                try_improve(best_x, 'Nelder-Mead', 2000, 1e-11)

    # Phase 4: Second round of restarts from improved best (cell finding insight)
    if time.time() < deadline - 20:
        np.random.seed(123)
        for k in range(10):
            if time.time() >= deadline - 15:
                break
            # Perturb best_x
            x0 = best_x + np.random.normal(0, 0.05, 6)
            x0 = np.clip(x0, [-1, 1e-3, -1, 1e-3, -1, 1e-3],
                         [1, 1.0, 1, 1.0, 1, 1.0])
            try_improve(x0, 'L-BFGS-B', 500, 1e-13)
            if time.time() < deadline - 10:
                try_improve(best_x, 'Nelder-Mead', 3000, 1e-12)

    # Phase 5: Final high-precision polish (refinement directive)
    if time.time() < deadline - 8:
        try_improve(best_x, 'Nelder-Mead', 10000, 1e-14)

    # Phase 6: Coordinate descent micro-polish
    if time.time() < deadline - 3:
        pts_cd = best_pts.copy()
        x_cd = best_x.copy()
        for it in range(3):
            if time.time() >= deadline - 2:
                break
            improved = False
            for i in range(6):
                for direction in [-1, 1]:
                    for step in [0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
                        if time.time() >= deadline - 1:
                            break
                        x_trial = x_cd.copy()
                        x_trial[i] += direction * step
                        # Check bounds
                        if i % 2 == 0:
                            if x_trial[i] < -1.0 or x_trial[i] > 1.0:
                                continue
                        else:
                            if x_trial[i] < 1e-3 or x_trial[i] > 1.0:
                                continue
                        pts_trial = build_points(x_trial)
                        diffs = pts_trial[:, np.newaxis, :] - pts_trial[np.newaxis, :, :]
                        dists = np.linalg.norm(diffs, axis=2)
                        D = np.max(dists)
                        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                        if D > 2.0 + 1e-8 or np.min(dists[mask]) < 1e-8:
                            continue
                        sc = score(pts_trial)
                        if sc > best_score:
                            best_score = sc
                            best_x = x_trial.copy()
                            best_pts = pts_trial
                            x_cd = x_trial.copy()
                            improved = True
            if not improved:
                break

    return best_pts.tolist()
