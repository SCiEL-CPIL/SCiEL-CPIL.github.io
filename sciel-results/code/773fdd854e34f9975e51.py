import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 140.0  # Evaluator kills at 150s, keep margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-8):
    # Bounds
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    # Circle in square
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    # Pairwise disjoint interiors
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    need = r[:, None] + r[None, :]
    # Check if any distance is strictly less than sum of radii
    # Use a small epsilon to account for floating point
    if np.any(dist < need - eps):
        return False
    return True

def _calc_max_radii(x, y, iters=100):
    """
    Calculate the maximum possible radii for given centers.
    r_i = min( dist_to_wall_i, min_{j!=i} (dist_ij - r_j) )
    We solve this system iteratively.
    """
    # Max radius based on walls
    r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Distances between centers
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    np.fill_diagonal(dist, np.inf)
    
    # Iterate to find consistent radii
    for _ in range(iters):
        # For each i, r_i <= dist_ij - r_j for all j
        # So r_i <= min_j (dist_ij - r_j)
        # New r_i = min(wall_limit_i, min_j(dist_ij - r_j))
        
        # Vectorized calculation of min_j(dist_ij - r_j)
        # dist_ij - r_j is a matrix. We want row-wise min.
        # dist - r[None, :]
        limits = dist - r[None, :]
        min_limits = np.min(limits, axis=1)
        
        # Update r
        new_r = np.minimum(r, min_limits)
        new_r = np.maximum(new_r, 0.0)
        
        if np.allclose(new_r, r, rtol=1e-8, atol=1e-10):
            r = new_r
            break
        r = new_r
        
    return r

def _sum_r(r):
    return float(np.sum(r))

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    r = _calc_max_radii(x, y)
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = _calc_max_radii(x, y)
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = _calc_max_radii(x, y)
    return x, y, r

def _objective(params):
    x = params[:N]
    y = params[N:2*N]
    r = _calc_max_radii(x, y)
    return -_sum_r(r)

def _optimize(x, y, start):
    best_x = x.copy()
    best_y = y.copy()
    best_r = _calc_max_radii(x, y)
    best_score = _sum_r(best_r)
    
    # Use Nelder-Mead for derivative-free optimization of centers
    # This handles the non-smooth nature of _calc_max_radii well
    try:
        res = minimize(
            _objective, 
            np.concatenate([x, y]), 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        vx = res.x[:N]
        vy = res.x[N:2*N]
        vr = _calc_max_radii(vx, vy)
        if _is_valid(vx, vy, vr):
            s = _sum_r(vr)
            if s > best_score:
                best_score = s
                best_x = vx
                best_y = vy
                best_r = vr
    except Exception:
        pass
        
    # Local perturbation search
    if _elapsed(start) < TIME_LIMIT - 10:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if _elapsed(start) > TIME_LIMIT - 10:
                break
            px = best_x + rng.uniform(-0.02, 0.02, N)
            py = best_y + rng.uniform(-0.02, 0.02, N)
            # Clamp to [0,1] roughly, _calc_max_radii will handle bounds
            px = np.clip(px, 0.01, 0.99)
            py = np.clip(py, 0.01, 0.99)
            
            pr = _calc_max_radii(px, py)
            if _is_valid(px, py, pr):
                s = _sum_r(pr)
                if s > best_score:
                    best_score = s
                    best_x = px
                    best_y = py
                    best_r = pr
                    # Re-optimize from this point if time permits
                    if _elapsed(start) < TIME_LIMIT - 20:
                         try:
                            res2 = minimize(
                                _objective, 
                                np.concatenate([px, py]), 
                                method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                            )
                            vx2 = res2.x[:N]
                            vy2 = res2.x[N:2*N]
                            vr2 = _calc_max_radii(vx2, vy2)
                            if _is_valid(vx2, vy2, vr2):
                                s2 = _sum_r(vr2)
                                if s2 > best_score:
                                    best_score = s2
                                    best_x = vx2
                                    best_y = vy2
                                    best_r = vr2
                         except Exception:
                             pass
                             
    return best_x, best_y, best_r

def construct():
    start = time.time()
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    
    best_x, best_y, best_r = inits[0]
    best_score = _sum_r(best_r)
    
    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 20:
            break
        res_x, res_y, res_r = _optimize(x, y, start)
        if _is_valid(res_x, res_y, res_r):
            s = _sum_r(res_r)
            if s > best_score:
                best_score = s
                best_x = res_x
                best_y = res_y
                best_r = res_r

    # Final safety shrink to ensure strict validity
    # The _is_valid check uses eps=1e-8.
    # We shrink by 1e-6 to be safe against evaluator's stricter checks.
    final_r = best_r * 0.99999
    
    # Verify final validity
    if not _is_valid(best_x, best_y, final_r):
        # If still invalid, shrink more
        final_r = best_r * 0.999
        if not _is_valid(best_x, best_y, final_r):
            final_r = best_r * 0.99
            if not _is_valid(best_x, best_y, final_r):
                # Last resort: tiny circles
                final_r = np.full(N, 0.01)
                # Re-center to be safe
                best_x = np.full(N, 0.5)
                best_y = np.full(N, 0.5)
                # This is a fallback, but unlikely to trigger if _is_valid works correctly

    result = np.column_stack((best_x, best_y, final_r))
    return result
