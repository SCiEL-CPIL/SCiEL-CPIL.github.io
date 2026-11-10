# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

_DEADLINE = 630.0
_TIME_MARGIN = 20.0  # safety margin before evaluator kill

# Precompute grid points
_XS = np.arange(-3.0, 1.5, 0.01)
_YS = np.arange(-1.5, 1.5, 0.01)
_X, _Y = np.meshgrid(_XS, _YS, indexing='ij')
_GX = _X.ravel()
_GY = _Y.ravel()
_NPTS = len(_GX)


def _densify_path(path):
    """Densify a path the same way the evaluator does."""
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)


def _check_in_base(u, v):
    """Check if points are inside the base L-shaped corridor."""
    arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return arm1 | arm2


def _evaluate_path_fast(path):
    """Fast evaluation with early termination. Returns count * 0.0001."""
    full = _densify_path(path)
    inside = np.ones(_NPTS, dtype=bool)
    for row in full:
        x, y, a = row[0], row[1], row[2]
        rad = -np.deg2rad(a)
        c = np.cos(rad)
        s = np.sin(rad)
        wx = _GX - x
        wy = _GY - y
        u = c * wx - s * wy
        v = s * wx + c * wy
        inside &= _check_in_base(u, v)
        cnt = inside.sum()
        if cnt == 0:
            return 0.0
    return cnt * 0.0001


def _evaluate_path(path):
    """Wrapper for compatibility."""
    return _evaluate_path_fast(path)


def _objective_neg(x):
    """Negative objective for scipy minimization."""
    n = len(x) // 3
    path = x.reshape(n, 3)
    # Enforce angle monotonicity and bounds
    path[:, 2] = np.clip(path[:, 2], 0.0, 90.0)
    path[:, 2] = np.maximum.accumulate(path[:, 2])
    return -_evaluate_path_fast(path)


def _local_search(path, n_iter=100, step_size=0.05, deadline=None):
    """Random local search with adaptive step size."""
    best_path = path.copy()
    best_val = _evaluate_path_fast(best_path)
    n = best_path.shape[0]
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - _TIME_MARGIN:
            break
        # Adaptive step: smaller when close to convergence
        cur_step = step_size * (1.0 - 0.5 * i / n_iter)
        trial = best_path + np.random.randn(n, 3) * cur_step
        trial[:, 2] = np.clip(trial[:, 2], 0.0, 90.0)
        trial[:, 2] = np.maximum.accumulate(trial[:, 2])
        val = _evaluate_path_fast(trial)
        if val > best_val:
            best_val = val
            best_path = trial
    
    return best_path, best_val


def _polish(path, deadline, n_iter=200):
    """High-precision polish using scipy Nelder-Mead with tight tolerance."""
    x0 = path.ravel()
    n = len(x0)
    
    def obj(x):
        p = x.reshape(-1, 3)
        p[:, 2] = np.clip(p[:, 2], 0.0, 90.0)
        p[:, 2] = np.maximum.accumulate(p[:, 2])
        return -_evaluate_path_fast(p)
    
    try:
        res = so.minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': n_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        best_x = res.x
    except Exception:
        best_x = x0
    
    best_path = best_x.reshape(-1, 3)
    best_path[:, 2] = np.clip(best_path[:, 2], 0.0, 90.0)
    best_path[:, 2] = np.maximum.accumulate(best_path[:, 2])
    return best_path


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Better initial path with more waypoints for smoother interpolation
    path = np.array(
        [
            [-0.05, 0.05, 10.0],
            [-0.10, 0.10, 18.0],
            [-0.18, 0.18, 26.0],
            [-0.25, 0.25, 34.0],
            [-0.30, 0.30, 42.0],
            [-0.35, 0.28, 50.0],
            [-0.42, 0.22, 58.0],
            [-0.50, 0.18, 66.0],
            [-0.58, 0.15, 74.0],
            [-0.65, 0.20, 82.0],
            [-0.72, 0.28, 88.0],
        ],
        dtype=np.float64,
    )
    
    best_path = path.copy()
    best_val = _evaluate_path_fast(best_path)
    
    # Phase 1: Coarse local search
    if time.time() < deadline - _TIME_MARGIN:
        best_path, best_val = _local_search(
            best_path, n_iter=80, step_size=0.15, deadline=deadline
        )
    
    # Phase 2: Medium local search
    if time.time() < deadline - _TIME_MARGIN:
        best_path, best_val = _local_search(
            best_path, n_iter=100, step_size=0.08, deadline=deadline
        )
    
    # Phase 3: Fine local search
    if time.time() < deadline - _TIME_MARGIN:
        best_path, best_val = _local_search(
            best_path, n_iter=100, step_size=0.03, deadline=deadline
        )
    
    # Phase 4: High-precision polish with scipy
    if time.time() < deadline - _TIME_MARGIN:
        polished = _polish(best_path, deadline, n_iter=300)
        pol_val = _evaluate_path_fast(polished)
        if pol_val > best_val:
            best_val = pol_val
            best_path = polished
    
    # Phase 5: Final micro-search around best
    if time.time() < deadline - _TIME_MARGIN:
        best_path, best_val = _local_search(
            best_path, n_iter=50, step_size=0.01, deadline=deadline
        )
    
    # Ensure valid output
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.shape[0] < 1:
        best_path = path
    if not np.all(np.isfinite(best_path)):
        best_path = path
    
    return best_path


# EVOLVE-BLOCK-END
