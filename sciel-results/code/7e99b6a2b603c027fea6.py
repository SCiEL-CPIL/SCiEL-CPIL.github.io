"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

_DEADLINE = 145.0  # safety margin before 150s kill


def _penalty(params, n, lam=1000.0):
    """Penalty for constraint violations."""
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    penalty = 0.0
    # Boundary constraints: circle must be inside [0,1]^2
    # Use linear penalty for boundaries to be smoother for DE
    penalty += np.sum(np.maximum(0, r - x))
    penalty += np.sum(np.maximum(0, r - (1 - x)))
    penalty += np.sum(np.maximum(0, r - y))
    penalty += np.sum(np.maximum(0, r - (1 - y)))
    # Non-overlap: distance >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            min_d = r[i] + r[j]
            if d < min_d:
                penalty += (min_d - d)
    return lam * penalty


def _total_objective(params, n, lam=1000.0):
    pts = params.reshape(n, 3)
    radii = pts[:, 2]
    return -np.sum(radii) + _penalty(params, n, lam)


def _is_valid(pts):
    """Check if configuration is valid (all constraints satisfied)."""
    n = len(pts)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(r <= 0):
        return False
    # Boundary
    if np.any(x - r < -1e-7) or np.any(1 - x - r < -1e-7) or np.any(y - r < -1e-7) or np.any(1 - y - r < -1e-7):
        return False
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _post_process(pts, n):
    """Enforce constraints by shrinking radii and adjusting positions."""
    pts = pts.copy()
    for _ in range(50):
        changed = False
        # Clip to boundaries
        for i in range(n):
            x, y, r = pts[i]
            r = min(r, x, 1 - x, y, 1 - y)
            if r < 0.001:
                r = 0.001
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            if abs(pts[i, 0] - x) > 1e-10 or abs(pts[i, 1] - y) > 1e-10 or abs(pts[i, 2] - r) > 1e-10:
                changed = True
            pts[i] = [x, y, r]
        
        # Resolve overlaps by shrinking
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2)
                min_d = pts[i, 2] + pts[j, 2]
                if d < min_d - 1e-7:
                    shrink = (min_d - d) / 2 + 1e-6
                    pts[i, 2] = max(0.001, pts[i, 2] - shrink)
                    pts[j, 2] = max(0.001, pts[j, 2] - shrink)
                    changed = True
        if not changed:
            break
    return pts


def _local_polish(params, n, time_limit=5.0):
    """Refine a solution using L-BFGS-B."""
    t0 = time.time()
    try:
        res = minimize(
            lambda p: _total_objective(p, n),
            params,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < _total_objective(params, n):
            return res.x
    except Exception:
        pass
    return params


def construct():
    n = N
    t_start = time.time()
    
    # Define bounds for DE
    # x, y in [0.01, 0.99], r in [0.01, 0.49]
    bounds = []
    for _ in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.01, 0.49))  # r
    
    best_pts = None
    best_score = -np.inf
    
    # Strategy 1: Differential Evolution (Global)
    # Use a relatively small population and iteration count to fit in time
    # popsize=15, maxiter=200 -> ~3000 evaluations.
    # Each eval is ~O(N^2) = 338 ops. Total ~1M ops. Very fast.
    try:
        if time.time() - t_start < _DEADLINE - 20:
            res_de = differential_evolution(
                lambda p: _total_objective(p, n),
                bounds=bounds,
                popsize=15,
                maxiter=200,
                seed=42,
                tol=1e-8,
                polish=False  # We will do our own polish
            )
            
            if res_de.success or res_de.fun < _total_objective(np.random.rand(n*3)*0.98+0.01, n):
                params = res_de.x
                # Local polish
                if time.time() - t_start < _DEADLINE - 5:
                    params = _local_polish(params, n, time_limit=10.0)
                
                pts = params.reshape(n, 3)
                pts = _post_process(pts, n)
                
                if _is_valid(pts):
                    score = np.sum(pts[:, 2])
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
    except Exception:
        pass
    
    # Strategy 2: Random Restarts with Local Polish (if time remains)
    if time.time() - t_start < _DEADLINE - 10:
        rng = np.random.RandomState(123)
        for i in range(20):
            if time.time() - t_start > _DEADLINE - 5:
                break
            
            # Random initialization
            params = np.array([
                [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.05, 0.2)]
                for _ in range(n)
            ]).flatten()
            
            # Local polish
            params = _local_polish(params, n, time_limit=2.0)
            pts = params.reshape(n, 3)
            pts = _post_process(pts, n)
            
            if _is_valid(pts):
                score = np.sum(pts[:, 2])
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
    
    # Fallback
    if best_pts is None or not _is_valid(best_pts):
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                pts.append([x, y, r])
        best_pts = _post_process(np.array(pts[:n]), n)
    
    return best_pts
# EVOLVE-BLOCK-END
