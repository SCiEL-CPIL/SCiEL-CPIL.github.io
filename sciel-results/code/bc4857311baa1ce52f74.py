"""Circle packing in unit square - 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

_DEADLINE = 115.0  # Safety margin before 120s kill

def _compute_radius_sum(res):
    return np.sum(res[:, 2])

def _is_valid(res, tol=1e-5):
    """Check if the packing is valid."""
    if len(res) != N:
        return False
    # Check radii are positive
    if np.any(res[:, 2] <= 0):
        return False
    # Check boundaries
    if np.any(res[:, 0] - res[:, 2] < -tol):
        return False
    if np.any(res[:, 1] - res[:, 2] < -tol):
        return False
    if np.any(1 - res[:, 0] - res[:, 2] < -tol):
        return False
    if np.any(1 - res[:, 1] - res[:, 2] < -tol):
        return False
    
    # Check pairwise distances
    # Vectorized check
    dx = res[:, 0][:, None] - res[:, 0][None, :]
    dy = res[:, 1][:, None] - res[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    r_sum = res[:, 2][:, None] + res[:, 2][None, :]
    # We need dist_sq >= r_sum^2 for i != j
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    if np.any(dist_sq[mask] < (r_sum**2)[mask] - tol):
        return False
    return True

def _fix_invalid(res, tol=1e-6):
    """Attempt to fix small violations by shrinking radii."""
    res = res.copy()
    # Shrink radii slightly
    res[:, 2] *= (1 - 1e-4)
    # If still invalid, shrink more aggressively
    if not _is_valid(res, tol=1e-4):
        res[:, 2] *= 0.9
    return res

def _vectorized_objective(params, n, lam=1000.0):
    """
    Vectorized objective function: negative sum of radii + penalty for violations.
    params: flat array of length 3*n: [x0, y0, r0, x1, y1, r1, ...]
    """
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Main objective: maximize sum of radii
    obj = -np.sum(r)
    
    penalty = 0.0
    
    # Boundary constraints: circle must be inside [0,1]^2
    # x - r >= 0, 1 - x - r >= 0, etc.
    # Violation is max(0, r - x) + max(0, r - (1-x)) ...
    # Squared penalty for smoothness
    b_pen = (
        np.sum(np.maximum(0, r - x))**2 +
        np.sum(np.maximum(0, r - (1 - x)))**2 +
        np.sum(np.maximum(0, r - y))**2 +
        np.sum(np.maximum(0, r - (1 - y)))**2
    )
    penalty += b_pen
    
    # Non-overlap: distance >= r_i + r_j
    # Vectorized pairwise distance calculation
    # x is (n,), y is (n,)
    # dx = x[:, None] - x[None, :] -> (n, n)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # min_dist matrix: r_i + r_j
    min_dist = r[:, None] + r[None, :]
    
    # We only care about i < j to avoid double counting and self-interaction
    # dist[i,i] = 0, min_dist[i,i] = 2*r_i. dist[i,i] < min_dist[i,i] is true for r>0.
    # So we must exclude diagonal.
    # Create a mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Get the distances and min distances for valid pairs
    d_pairs = dist[mask]
    min_d_pairs = min_dist[mask]
    
    # Violation: min_d - d if d < min_d
    violations = np.maximum(0, min_d_pairs - d_pairs)
    penalty += np.sum(violations**2)
    
    # Radii must be positive
    penalty += np.sum(np.maximum(0, -r))**2
    
    return obj + lam * penalty

def _post_process(pts, n):
    """Enforce constraints strictly by shrinking radii if necessary."""
    pts = pts.copy()
    # First, ensure boundaries
    for i in range(n):
        x, y, r = pts[i]
        r = min(r, x, 1 - x, y, 1 - y)
        if r < 1e-6:
            r = 1e-6
        pts[i, 0] = np.clip(x, r, 1 - r)
        pts[i, 1] = np.clip(y, r, 1 - r)
        pts[i, 2] = r
        
    # Resolve overlaps by shrinking
    # Iterate a few times
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pts[i, 0] - pts[j, 0])**2 + (pts[i, 1] - pts[j, 1])**2)
                min_d = pts[i, 2] + pts[j, 2]
                if d < min_d - 1e-10:
                    # Shrink both radii by half the overlap + epsilon
                    shrink = (min_d - d) / 2.0 + 1e-7
                    pts[i, 2] = max(1e-6, pts[i, 2] - shrink)
                    pts[j, 2] = max(1e-6, pts[j, 2] - shrink)
                    # Adjust positions if they are too close to boundary after shrink?
                    # Actually, shrinking radius makes boundary constraint easier to satisfy.
                    changed = True
        if not changed:
            break
            
    return pts

def _make_grid_initial(n):
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            pts.append([x, y, r])
    return np.array(pts[:n])

def _make_hex_initial(n):
    # Hexagonal packing pattern
    pts = []
    # Try different row counts to fit n
    for rows in range(3, 10):
        cols = int(np.ceil(n / rows))
        if cols < 2:
            continue
        dx = 1.0 / (cols + 0.5)
        dy = 1.0 / (rows + 0.5)
        r = min(dx, dy) * 0.48
        count = 0
        for row in range(rows):
            offset = 0.0 if row % 2 == 0 else dx / 2
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5 + offset) * dx
                y = (row + 0.5) * dy
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                pts.append([x, y, r])
                count += 1
        if len(pts) >= n:
            break
    if len(pts) < n:
        return _make_grid_initial(n)
    return np.array(pts[:n])

def _make_random_initial(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n):
        r = rng.uniform(0.02, 0.12)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        pts.append([x, y, r])
    return np.array(pts)

def _optimize_single(pts0, n, time_limit=5.0):
    """Run optimization on a single initial configuration."""
    params = pts0.flatten().copy()
    best_params = params.copy()
    best_obj = _vectorized_objective(params, n, lam=500.0)
    
    t0 = time.time()
    
    # Stage 1: L-BFGS-B
    try:
        res = minimize(
            lambda p: _vectorized_objective(p, n, lam=500.0),
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    if time.time() - t0 > time_limit * 0.5:
        return best_params.reshape(n, 3)
    
    # Stage 2: Nelder-Mead for fine tuning
    try:
        res2 = minimize(
            lambda p: _vectorized_objective(p, n, lam=1000.0),
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
        )
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
        
    return best_params.reshape(n, 3)

def construct():
    n = N
    t_start = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _make_grid_initial(n),
        _make_hex_initial(n),
        _make_random_initial(n, seed=1),
        _make_random_initial(n, seed=2),
        _make_random_initial(n, seed=3),
        _make_random_initial(n, seed=4),
        _make_random_initial(n, seed=5),
        _make_random_initial(n, seed=6),
        _make_random_initial(n, seed=7),
        _make_random_initial(n, seed=8),
    ]
    
    best_pts = None
    best_score = -np.inf
    
    for idx, init in enumerate(inits):
        elapsed = time.time() - t_start
        if elapsed > _DEADLINE - 5:
            break
        
        remaining = _DEADLINE - elapsed - 5
        if remaining < 1:
            break
        
        # Allocate time per restart
        time_limit = min(5.0, remaining / max(1, len(inits) - idx))
        
        try:
            opt_pts = _optimize_single(init, n, time_limit=time_limit)
            opt_pts = _post_process(opt_pts, n)
            
            if _is_valid(opt_pts):
                score = np.sum(opt_pts[:, 2])
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # If no valid solution found, fall back to grid
    if best_pts is None or not _is_valid(best_pts):
        best_pts = _make_grid_initial(n)
        best_pts = _post_process(best_pts, n)
        
    return best_pts
# EVOLVE-BLOCK-END
