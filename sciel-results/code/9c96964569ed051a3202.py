# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

TIME_LIMIT = 610.0
START_TIME = time.time()


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _sofa_points():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return gx.ravel(), gy.ravel()


def _score_path_fast(path, gx, gy):
    """
    Score a path: the sofa is the set of points inside EVERY placement.
    path: (n, 3) array of (x, y, a)
    Returns score = count * 0.0001
    """
    n_points = len(gx)
    mask = np.ones(n_points, dtype=bool)
    
    for i in range(len(path)):
        x, y, a = path[i]
        a_rad = np.deg2rad(a)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        u = ca * (gx - x) + sa * (gy - y)
        v = -sa * (gx - x) + ca * (gy - y)
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        inside = in_rect1 | in_rect2
        mask &= inside
        if not np.any(mask):
            return 0.0
    
    return float(np.sum(mask)) * 0.0001


def _evaluate_full_path(path, gx, gy):
    """
    Prepend (0,0,0), append (-2,0,90), densify, then score.
    """
    full = np.vstack([
        np.array([0.0, 0.0, 0.0]),
        path,
        np.array([-2.0, 0.0, 90.0])
    ])
    full = _densify_path(full)
    return _score_path_fast(full, gx, gy)


def _densify_path(path):
    """
    Insert linear interpolations between consecutive rows.
    m = max(1, ceil(|a2 - a1|)) equal steps.
    """
    result = [path[0].copy()]
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        m = max(1, int(np.ceil(abs(p2[2] - p1[2]))))
        for k in range(1, m):
            t = k / m
            pt = p1 + t * (p2 - p1)
            result.append(pt)
        result.append(p2)
    return np.array(result)


def _objective_de(params, gx, gy, n_steps):
    """
    Objective for differential evolution.
    params: 2*n_steps array of [x0,y0,x1,y1,...,xn-1,yn-1]
    Angles are fixed at evenly spaced values from 1 to 89.
    """
    angles = np.linspace(1.0, 89.0, n_steps)
    path = np.column_stack([params[0::2], params[1::2], angles])
    path[:, 0] = np.clip(path[:, 0], -3.0, 3.0)
    path[:, 1] = np.clip(path[:, 1], -3.0, 3.0)
    score = _evaluate_full_path(path, gx, gy)
    return -score


def _objective_lbfgsb(params, gx, gy):
    """
    Objective for L-BFGS-B minimization.
    params: 3*n array of [x0,y0,a0, x1,y1,a1, ...]
    """
    n = len(params) // 3
    path = params.reshape(n, 3)
    path[:, 2] = np.clip(path[:, 2], 1.0, 89.0)
    for i in range(1, n):
        if path[i, 2] <= path[i - 1, 2]:
            path[i, 2] = path[i - 1, 2] + 0.1
    path[:, 0] = np.clip(path[:, 0], -3.0, 3.0)
    path[:, 1] = np.clip(path[:, 1], -3.0, 3.0)
    score = _evaluate_full_path(path, gx, gy)
    return -score


def _bounds_lbfgsb(n):
    """Bounds for L-BFGS-B: x in [-3,3], y in [-3,3], a in [1,89]"""
    bounds = []
    for _ in range(n):
        bounds.append((-3.0, 3.0))
        bounds.append((-3.0, 3.0))
        bounds.append((1.0, 89.0))
    return bounds


def _local_polish(path, gx, gy, n_iter=50):
    """
    Local coordinate ascent polish on the path.
    """
    best_path = path.copy()
    best_score = _evaluate_full_path(best_path, gx, gy)
    n_points = len(best_path)
    
    for _ in range(n_iter):
        if _remaining_time() < 15:
            break
        improved = False
        for i in range(n_points):
            for dim in range(3):
                old_val = best_path[i, dim]
                for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3]:
                    test_path = best_path.copy()
                    test_path[i, dim] = old_val + delta
                    test_path[:, 2] = np.clip(test_path[:, 2], 1.0, 89.0)
                    for j in range(1, n_points):
                        if test_path[j, 2] <= test_path[j - 1, 2]:
                            test_path[j, 2] = test_path[j - 1, 2] + 0.1
                    test_path[:, 0] = np.clip(test_path[:, 0], -3.0, 3.0)
                    test_path[:, 1] = np.clip(test_path[:, 1], -3.0, 3.0)
                    
                    test_score = _evaluate_full_path(test_path, gx, gy)
                    if test_score > best_score + 1e-8:
                        best_score = test_score
                        best_path = test_path.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_path, best_score


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path that maximizes the area of the sofa that can be moved through the L-corridor.
    Combines differential evolution (global) with L-BFGS-B (local refinement) and coordinate polish.
    """
    gx, gy = _sofa_points()
    
    # Baseline path
    baseline = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_path = baseline.copy()
    best_score = _evaluate_full_path(best_path, gx, gy)
    
    n_steps = 8
    
    # Phase 1: Differential evolution for global exploration
    np.random.seed(42)
    
    try:
        if _remaining_time() > 120:
            de_result = differential_evolution(
                _objective_de,
                [(-3.0, 3.0)] * (2 * n_steps),
                args=(gx, gy, n_steps),
                seed=42,
                maxiter=40,
                popsize=15,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True,
                init='latinhypercube'
            )
            
            if de_result.success and _remaining_time() > 30:
                de_params = de_result.x
                angles = np.linspace(1.0, 89.0, n_steps)
                de_path = np.column_stack([de_params[0::2], de_params[1::2], angles])
                de_path[:, 0] = np.clip(de_path[:, 0], -3.0, 3.0)
                de_path[:, 1] = np.clip(de_path[:, 1], -3.0, 3.0)
                de_score = _evaluate_full_path(de_path, gx, gy)
                
                if de_score > best_score:
                    best_score = de_score
                    best_path = de_path.copy()
                    print(f"DE improved to {best_score:.6f}")
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Phase 2: L-BFGS-B refinement on all 3 dimensions (x, y, a)
    if _remaining_time() > 60:
        try:
            # Start from current best
            x0 = best_path.ravel()
            res = minimize(
                _objective_lbfgsb, x0, args=(gx, gy),
                method='L-BFGS-B',
                bounds=_bounds_lbfgsb(n_steps),
                options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if _remaining_time() > 25:
                res_path = res.x.reshape(n_steps, 3)
                res_path[:, 2] = np.clip(res_path[:, 2], 1.0, 89.0)
                for i in range(1, n_steps):
                    if res_path[i, 2] <= res_path[i - 1, 2]:
                        res_path[i, 2] = res_path[i - 1, 2] + 0.1
                res_path[:, 0] = np.clip(res_path[:, 0], -3.0, 3.0)
                res_path[:, 1] = np.clip(res_path[:, 1], -3.0, 3.0)
                res_score = _evaluate_full_path(res_path, gx, gy)
                
                if res_score > best_score:
                    best_score = res_score
                    best_path = res_path.copy()
                    print(f"L-BFGS-B improved to {best_score:.6f}")
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")
    
    # Phase 3: Local coordinate polish
    if _remaining_time() > 30:
        try:
            polished_path, polished_score = _local_polish(best_path, gx, gy, n_iter=50)
            if polished_score > best_score:
                best_score = polished_score
                best_path = polished_path.copy()
                print(f"Polish improved to {best_score:.6f}")
        except Exception as e:
            print(f"Polish failed: {e}")
    
    return best_path


# EVOLVE-BLOCK-END
