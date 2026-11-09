import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0

    def compute_dists(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists

    def get_unique_dists(dists):
        iu = np.triu_indices(8, k=1)
        return dists[iu]

    def compute_score_log(points):
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        if np.min(d_pairs) < 1e-10:
            return -np.inf
        log_prod = np.sum(np.log(d_pairs))
        D = np.max(d_pairs)
        log_D = np.log(D)
        score_log = log_prod - 28.0 * log_D
        return score_log

    def objective(params):
        points = params.reshape(8, 2)
        score_log = compute_score_log(points)
        if np.isinf(score_log):
            return 1e10
        return -score_log

    # Symmetric parameterization:
    # 4 points at angles: a, pi/2 + a, pi + a, 3*pi/2 + a  (radius r1)
    # 4 points at angles: pi/4 + a, 3*pi/4 + a, 5*pi/4 + a, 7*pi/4 + a  (radius r2)
    # This preserves D4 symmetry (symmetric under 90-degree rotation and reflection)
    # Parameters: [r1, r2, a]
    
    def symmetric_points(params):
        r1, r2, a = params
        if r1 < 0: r1 = 0.0
        if r2 < 0: r2 = 0.0
        # Ensure distinctness
        if r1 < 1e-6 and r2 < 1e-6:
            r1 = 1.0
            r2 = 1.0
        angles1 = a + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = a + np.pi/4 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        points = np.vstack((pts1, pts2))
        return points

    def objective_sym(params):
        points = symmetric_points(params)
        score_log = compute_score_log(points)
        if np.isinf(score_log):
            return 1e10
        return -score_log

    def objective_full(params):
        points = params.reshape(8, 2)
        score_log = compute_score_log(points)
        if np.isinf(score_log):
            return 1e10
        return -score_log

    def polish(points, method='Nelder-Mead'):
        def obj(params):
            p = params.reshape(8, 2)
            sl = compute_score_log(p)
            if np.isinf(sl):
                return 1e10
            return -sl
        res = minimize(obj, points.flatten(), method=method,
                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12, 'niter': 50000})
        if res.fun < 1e9:
            return res.x.reshape(8, 2)
        return points

    best_points = None
    best_score = -np.inf

    # Strategy 1: Optimize within D4 symmetric family
    # Initial guesses for symmetric parameters
    sym_inits = []
    # Regular octagon: r1 = r2 = 1, a = 0
    sym_inits.append([1.0, 1.0, 0.0])
    # Various ratios
    for r1 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        for r2 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            for a in [0.0, np.pi/8, np.pi/16, -np.pi/8, -np.pi/16]:
                sym_inits.append([r1, r2, a])

    for init in sym_inits:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
        try:
            res = minimize(objective_sym, init, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'niter': 20000})
            if res.fun < 1e9:
                pts = symmetric_points(res.x)
                sl = compute_score_log(pts)
                if sl > best_score:
                    best_score = sl
                    best_points = pts
        except Exception:
            pass

    # Strategy 2: Full optimization from good starting points
    if best_points is not None:
        # Use the best symmetric point as a starting point for full optimization
        full_inits = [best_points]
        # Add some perturbations
        rng = np.random.default_rng(123)
        for _ in range(5):
            pert = rng.normal(0, 0.05, size=best_points.shape)
            full_inits.append(best_points + pert)
        # Add the regular octagon
        angles = np.arange(8) * (np.pi / 4)
        oct_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        full_inits.append(oct_pts)

        for init in full_inits:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 15:
                break
            try:
                polished = polish(init, method='Nelder-Mead')
                sl = compute_score_log(polished)
                if sl > best_score:
                    best_score = sl
                    best_points = polished
            except Exception:
                pass

    # Final attempt with different method
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 10:
            try:
                def obj(params):
                    p = params.reshape(8, 2)
                    sl = compute_score_log(p)
                    if np.isinf(sl):
                        return 1e10
                    return -sl
                res = minimize(obj, best_points.flatten(), method='Powell',
                               options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12, 'niter': 10000})
                if res.fun < 1e9:
                    pts = res.x.reshape(8, 2)
                    sl = compute_score_log(pts)
                    if sl > best_score:
                        best_score = sl
                        best_points = pts
            except Exception:
                pass

    if best_points is None:
        angles = np.arange(8) * (np.pi / 4)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    # Normalize to diameter 2
    dists = compute_dists(best_points)
    D = np.max(dists)
    if D > 0:
        best_points = best_points * (2.0 / D)

    # Verify
    dists = compute_dists(best_points)
    iu = np.triu_indices(8, k=1)
    if np.min(dists[iu]) < 1e-6 or np.any(np.abs(best_points) >= 1e6):
        angles = np.arange(8) * (np.pi / 4)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_points = best_points * (2.0 / (2 * np.cos(np.pi/8)))

    return best_points.tolist()
