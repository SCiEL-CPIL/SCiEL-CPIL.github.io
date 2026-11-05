"""Optimize 8 planar points to maximize product of pairwise distances at fixed diameter."""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _log_score(points):
    """Compute log of Delta/n^n where Delta is product of distances over ordered pairs, after scaling to diameter 2."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return -np.inf
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    log_score = log_delta - n * np.log(n)
    return log_score

def _score(points):
    """Compute Delta/n^n (raw score)."""
    ls = _log_score(points)
    if not np.isfinite(ls):
        return -np.inf
    if ls > 50:
        return float('inf')
    return math.exp(ls)

def _neg_log_score(params):
    """Negative log of the score for minimization. Returns large value if invalid."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-10:
                return 1e10
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls

def _neg_smooth_log_score(params, beta):
    """Smooth surrogate using soft-max for diameter, for gradient-based methods."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return 1e10
    max_d = np.max(valid_dists)
    if max_d <= 0:
        return 1e10
    shifted = beta * (valid_dists - max_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_log_diameter = max_d + log_sum_exp / beta
    n_pairs = 8 * 7
    log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * smooth_log_diameter
    log_score = log_delta - 8 * np.log(8)
    if not np.isfinite(log_score):
        return 1e10
    return -log_score

def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _two_concentric_squares(r1, r2, rot1, rot2):
    pts = []
    for k in range(4):
        angle = rot1 + 2 * np.pi * k / 4
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    for k in range(4):
        angle = rot2 + 2 * np.pi * k / 4
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)

def _random_initial(rng, scale=1.0):
    angles = 2 * np.pi * np.arange(8) / 8 + rng.normal(0, 0.3, 8)
    r = 1.0 + rng.normal(0, 0.2, 8)
    r = np.abs(r) + 0.1
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0

    best_points = _regular_octagon()
    best_score = _score(best_points)
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(42)

    # List of initial configurations to try
    initial_configs = []

    # 1. Regular octagon
    initial_configs.append(_regular_octagon())

    # 2. Two concentric squares with varied parameters
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            if abs(r1 - r2) < 0.1:
                continue
            for rot1 in [0.0, np.pi/8]:
                for rot2 in [0.0, np.pi/8]:
                    initial_configs.append(_two_concentric_squares(r1, r2, rot1, rot2))

    # 3. Random perturbations of the octagon
    for _ in range(30):
        initial_configs.append(_random_initial(rng))

    # 4. Antipodal pairs
    for _ in range(15):
        base_angles = rng.uniform(0, np.pi/2, 4)
        radii = rng.uniform(0.5, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r * np.cos(a), r * np.sin(a)])
            pts.append([-r * np.cos(a), -r * np.sin(a)])
        initial_configs.append(np.array(pts))

    # 5. Random points
    for _ in range(30):
        radii = rng.uniform(0.1, 1.8, 8)
        angles_r = rng.uniform(0, 2 * np.pi, 8)
        initial_configs.append(np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)]))

    # 6. Rectangular grid perturbations
    for _ in range(15):
        xs = rng.uniform(-1, 1, 8)
        ys = rng.uniform(-1, 1, 8)
        initial_configs.append(np.column_stack([xs, ys]))

    # Score and sort initializations
    scored_inits = []
    for init_p in initial_configs:
        if time.time() > _DEADLINE - 60:
            break
        try:
            ls = _log_score(init_p)
            if np.isfinite(ls):
                scored_inits.append((ls, init_p))
        except Exception:
            pass

    scored_inits.sort(key=lambda x: x[0], reverse=True)

    # Optimization pass 1: Nelder-Mead on top initializations
    for ls, init_p in scored_inits[:25]:
        if time.time() > _DEADLINE - 30:
            break
        x0 = init_p.flatten()
        try:
            res = minimize(_neg_log_score, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < 1e10:
                optimal_points = res.x.reshape(8, 2)
                ls_val = -res.fun
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    sc = math.exp(ls_val) if ls_val < 50 else float('inf')
                    if sc > best_score:
                        best_score = sc
                        best_log_score = ls_val
                        best_points = optimal_points
        except Exception:
            pass

    # Optimization pass 2: L-BFGS-B with smooth surrogate for top initializations
    if time.time() < _DEADLINE - 25:
        for ls, init_p in scored_inits[:15]:
            if time.time() > _DEADLINE - 20:
                break
            x0 = init_p.flatten()
            for beta in [50.0, 100.0, 200.0]:
                if time.time() > _DEADLINE - 15:
                    break
                try:
                    res = minimize(lambda x: _neg_smooth_log_score(x, beta), x0,
                                   method='L-BFGS-B',
                                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                    if res.fun < 1e10:
                        optimal_points = res.x.reshape(8, 2)
                        ls_val = _log_score(optimal_points)
                        if np.isfinite(ls_val) and ls_val > best_log_score:
                            sc = math.exp(ls_val) if ls_val < 50 else float('inf')
                            if sc > best_score:
                                best_score = sc
                                best_log_score = ls_val
                                best_points = optimal_points
                except Exception:
                    pass

    # Optimization pass 3: COBYLA on top initializations
    if time.time() < _DEADLINE - 20:
        for ls, init_p in scored_inits[:10]:
            if time.time() > _DEADLINE - 15:
                break
            x0 = init_p.flatten()
            try:
                res = minimize(_neg_log_score, x0, method='COBYLA',
                               options={'maxiter': 3000, 'rhobeg': 0.1})
                if res.fun < 1e10:
                    optimal_points = res.x.reshape(8, 2)
                    ls_val = -res.fun
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        sc = math.exp(ls_val) if ls_val < 50 else float('inf')
                        if sc > best_score:
                            best_score = sc
                            best_log_score = ls_val
                            best_points = optimal_points
            except Exception:
                pass

    # High-precision polish: refine best_points with increasing beta
    if time.time() < _DEADLINE - 10:
        current_best = best_points.copy()
        for beta in [100.0, 500.0, 1000.0]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                res = minimize(lambda x: _neg_smooth_log_score(x, beta), current_best.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e10:
                    refined = res.x.reshape(8, 2)
                    ls_val = _log_score(refined)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        sc = math.exp(ls_val) if ls_val < 50 else float('inf')
                        if sc > best_score:
                            best_score = sc
                            best_log_score = ls_val
                            best_points = refined
                            current_best = refined.copy()
            except Exception:
                pass

        # Final Nelder-Mead polish on the best
        if time.time() < _DEADLINE - 5:
            try:
                res = minimize(_neg_log_score, best_points.flatten(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < 1e10:
                    refined = res.x.reshape(8, 2)
                    ls_val = _log_score(refined)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        sc = math.exp(ls_val) if ls_val < 50 else float('inf')
                        if sc > best_score:
                            best_score = sc
                            best_log_score = ls_val
                            best_points = refined
            except Exception:
                pass

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
