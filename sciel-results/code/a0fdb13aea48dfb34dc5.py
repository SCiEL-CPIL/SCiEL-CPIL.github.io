"""Circle packing in unit square - vectorized force-directed optimization."""
N = 26

import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0
_start_time = None


def _hexagonal_init(n, jitter=0.0, rng=None):
    """Initialize n circles on a hexagonal lattice pattern."""
    if rng is None:
        rng = np.random
    
    side = int(np.ceil(np.sqrt(n)))
    s = 1.0 / (side + 0.5)
    row_h = s * np.sqrt(3) / 2
    
    positions = []
    row = 0
    while len(positions) < n and row < 3 * side:
        cols_in_row = side if row % 2 == 0 else side - 1
        for col in range(cols_in_row):
            x = (col + 0.5) * s + (row % 2) * s / 2
            y = row * row_h + s * 0.5
            if 0.01 <= x <= 0.99 and 0.01 <= y <= 0.99:
                positions.append((x, y))
        row += 1
    
    while len(positions) < n:
        positions.append((0.5 + 0.1 * rng.randn(), 0.5 + 0.1 * rng.randn()))
    
    positions = np.array(positions[:n])
    
    if jitter > 0:
        positions = positions + jitter * rng.randn(n, 2)
    
    positions = np.clip(positions, 0.01, 0.99)
    
    r0 = 0.03
    return positions, np.full(n, r0)


def _vec_objective(params, n, lam):
    """Vectorized objective: -sum(r) + lam * boundary_violations + lam * pairwise_violations"""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    obj = -np.sum(r)
    
    # Boundary violations (vectorized)
    v_left = np.maximum(0.0, r - x)
    v_right = np.maximum(0.0, r - (1.0 - x))
    v_bottom = np.maximum(0.0, r - y)
    v_top = np.maximum(0.0, r - (1.0 - y))
    
    obj += lam * (np.sum(v_left**2) + np.sum(v_right**2) + 
                  np.sum(v_bottom**2) + np.sum(v_top**2))
    
    # Pairwise violations (vectorized)
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    r_sum = r[:, None] + r[None, :]
    
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    v = np.maximum(0.0, r_sum[iu] - dist[iu])
    obj += lam * np.sum(v**2)
    
    return obj


def _vec_gradient(params, n, lam):
    """Vectorized gradient."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    grad = np.zeros_like(params)
    grad[2*n:] = -1.0
    
    # Boundary gradients
    v_left = np.maximum(0.0, r - x)
    v_right = np.maximum(0.0, r - (1.0 - x))
    v_bottom = np.maximum(0.0, r - y)
    v_top = np.maximum(0.0, r - (1.0 - y))
    
    mask_l = v_left > 0
    mask_r = v_right > 0
    mask_b = v_bottom > 0
    mask_t = v_top > 0
    
    if np.any(mask_l):
        grad[2*n:][mask_l] += lam * 2 * v_left[mask_l]
        grad[0::2][mask_l] += -lam * 2 * v_left[mask_l]
    
    if np.any(mask_r):
        grad[2*n:][mask_r] += lam * 2 * v_right[mask_r]
        grad[0::2][mask_r] += lam * 2 * v_right[mask_r]
    
    if np.any(mask_b):
        grad[2*n:][mask_b] += lam * 2 * v_bottom[mask_b]
        grad[1::2][mask_b] += -lam * 2 * v_bottom[mask_b]
    
    if np.any(mask_t):
        grad[2*n:][mask_t] += lam * 2 * v_top[mask_t]
        grad[1::2][mask_t] += lam * 2 * v_top[mask_t]
    
    # Pairwise gradients
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(n, k=1)
    v = np.maximum(0.0, r_sum[iu] - dist[iu])
    mask = v > 0
    
    if np.any(mask):
        ii = iu[0][mask]
        jj = iu[1][mask]
        vv = v[mask]
        ddist = dist[iu][mask]
        ddx = dx[iu][mask]
        ddy = dy[iu][mask]
        
        factor = lam * 2 * vv / ddist
        
        # dx_i, dy_i, dx_j, dy_j
        grad[2*ii] += -factor * ddx
        grad[2*ii + 1] += -factor * ddy
        grad[2*jj] += factor * ddx
        grad[2*jj + 1] += factor * ddy
        
        # dr_i, dr_j
        grad[2*n + ii] += lam * 2 * vv
        grad[2*n + jj] += lam * 2 * vv
    
    return grad


def _check_valid(centers, radii, tol=1e-7):
    """Check validity."""
    x, y = centers[:, 0], centers[:, 1]
    r = radii
    
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(len(r), k=1)
    if np.any(dist[iu] < r_sum[iu] - tol):
        return False
    return True


def _repair(centers, radii, max_iter=200):
    """Repair small violations."""
    x = centers[:, 0].copy()
    y = centers[:, 1].copy()
    r = radii.copy()
    n = len(r)
    
    # Clip to boundary
    for i in range(n):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i], 0.5)
        r[i] = max(r[i], 0.0)
    
    for _ in range(max_iter):
        changed = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(n, k=1)
        overlaps = dist[iu] < r_sum[iu] - 1e-10
        
        if not np.any(overlaps):
            break
        
        ii = iu[0][overlaps]
        jj = iu[1][overlaps]
        excess = (r_sum[iu][overlaps] - dist[iu][overlaps])
        
        for k in range(len(ii)):
            i, j = ii[k], jj[k]
            e = excess[k]
            if e > 0:
                r[i] -= e / 2
                r[j] -= e / 2
                r[i] = max(0.0, min(r[i], x[i], 1 - x[i], y[i], 1 - y[i]))
                r[j] = max(0.0, min(r[j], x[j], 1 - x[j], y[j], 1 - y[j]))
                changed = True
        
        if not changed:
            break
    
    return np.column_stack([x, y]), r


def construct():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = N
    best_score = -np.inf
    best_params = None
    
    rng = np.random.RandomState(42)
    
    bounds = [(0.001, 0.999)] * (2 * n) + [(0.001, 0.5)] * n
    
    for restart in range(30):
        elapsed = time.time() - _start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Initialize
        jitter = 0.0 if restart < 2 else 0.02 + 0.01 * rng.rand()
        centers, radii = _hexagonal_init(n, jitter=jitter, rng=rng)
        params0 = np.concatenate([centers.ravel(), radii])
        
        best_local = None
        best_local_score = -np.inf
        
        # Multi-stage lambda increase
        for lam in [50.0, 200.0, 1000.0, 5000.0]:
            elapsed = time.time() - _start_time
            if elapsed > DEADLINE - 3:
                break
            
            try:
                result = minimize(
                    _vec_objective,
                    params0,
                    args=(n, lam),
                    jac=_vec_gradient,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-9, 'gtol': 1e-6}
                )
                params0 = result.x
            except Exception:
                break
            
            c = params0[:2*n].reshape(n, 2)
            r = params0[2*n:]
            
            c, r = _repair(c, r)
            
            if _check_valid(c, r):
                score = np.sum(r)
                if score > best_local_score:
                    best_local_score = score
                    best_local = np.column_stack([c[:, 0], c[:, 1], r])
            
            params0 = np.concatenate([c.ravel(), r])
        
        if best_local is not None and best_local_score > best_score:
            best_score = best_local_score
            best_params = best_local
    
    if best_params is not None:
        return best_params
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
