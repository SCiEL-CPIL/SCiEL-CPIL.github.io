"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START) - 3.0

def _max_radius_for_positions(pos):
    """Vectorized: given positions (N,2), compute max radius for each circle."""
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)

def _objective(pos):
    return float(np.sum(_max_radius_for_positions(pos)))

def _is_valid(pos, eps=1e-9):
    """Check validity with computed radii."""
    radii = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < radii[i] + radii[j] - eps:
                return False
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1.0 - radii[i] + eps or
                pos[i, 1] < radii[i] - eps or pos[i, 1] > 1.0 - radii[i] + eps):
            return False
    return True

def _soft_objective(z, lam):
    """
    z = [x0,y0,r0, x1,y1,r1, ...]. Minimize -sum(r) + lam*(overlap + wall penalties).
    Uses quadratic penalties for smoothness.
    Vectorized pairwise calculations.
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    # wall penalties
    pen += np.sum(np.maximum(r - x, 0.0) ** 2)
    pen += np.sum(np.maximum(x + r - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(r - y, 0.0) ** 2)
    pen += np.sum(np.maximum(y + r - 1.0, 0.0) ** 2)
    
    # pairwise overlap vectorized
    # Compute distance matrix
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    
    needs = r[:, None] + r[None, :]
    # Only consider pairs i < j to avoid double counting in penalty sum?
    # Actually, penalty is sum over all pairs. 
    # The penalty term (need - d)^2 is symmetric.
    # If we sum over all i,j, we count each pair twice.
    # The gradient will also be doubled. This is fine as long as consistent.
    # However, standard practice is to sum over unique pairs.
    # Let's stick to unique pairs for clarity and to match gradient logic if we write it explicitly.
    # But vectorizing unique pairs is harder. 
    # Let's just sum over all i!=j and divide by 2? Or just sum all and let gradient handle it?
    # If objective is 0.5 * sum_{i!=j} (need-d)^2, then grad is sum over neighbors.
    # Let's use 0.5 factor in objective to make gradient cleaner.
    
    diff = needs - d
    diff[diff < 0] = 0.0
    np.fill_diagonal(diff, 0.0)
    
    # Sum of squares of positive differences
    # This counts each pair twice.
    pen += 0.5 * np.sum(diff**2)
    
    return base + lam * pen

def _soft_grad(z, lam):
    """Analytic gradient of _soft_objective."""
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    gr -= 1.0  # d(-sum r)/dr = -1
    
    # Wall gradients
    # m = r - x, pen = m^2. d/dx = -2m, d/dr = 2m. (if m>0)
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
        
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
        
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
        
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]

    # Pairwise gradients
    # Objective term: 0.5 * sum_{i!=j} (max(0, r_i + r_j - d_ij))^2
    # Let f_ij = max(0, r_i + r_j - d_ij)
    # d/dx_i sum_{j!=i} 0.5 * f_ij^2 = sum_{j!=i} f_ij * d f_ij / d x_i
    # d f_ij / d x_i = - d d_ij / d x_i (if active)
    # d d_ij / d x_i = (x_i - x_j) / d_ij
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    
    needs = r[:, None] + r[None, :]
    diff = needs - d
    diff[diff < 0] = 0.0
    np.fill_diagonal(diff, 0.0)
    
    # We need sum over j for each i of diff[i,j] * (-(x_i - x_j)/d_ij)
    # Note: diff is symmetric. d is symmetric.
    # grad_x_i = - sum_{j!=i} diff[i,j] * dx[i,j] / d[i,j]
    
    # Avoid division by zero (diagonal is inf, so 1/d is 0)
    inv_d = np.zeros_like(d)
    mask = d > 1e-12
    inv_d[mask] = 1.0 / d[mask]
    
    # Term for x: - diff * dx * inv_d
    term_x = -diff * dx * inv_d
    term_y = -diff * dy * inv_d
    
    # Sum over j (axis 1) for each i
    gx += lam * 2.0 * np.sum(term_x, axis=1)
    gy += lam * 2.0 * np.sum(term_y, axis=1)
    
    # Gradient for r_i: sum_{j!=i} diff[i,j] * 1
    gr += lam * 2.0 * np.sum(diff, axis=1)
    
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out

def _project_feasible(pos, rad, iters=30):
    """Repair: clamp to walls, shrink overlapping radii. Returns (pos, rad, feasible)."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    for _ in range(iters):
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            r = min(r, x, 1.0 - x, y, 1.0 - y)
            r = max(r, 0.0)
            x = min(max(x, r), 1.0 - r)
            y = min(max(y, r), 1.0 - r)
            pos[i, 0], pos[i, 1], rad[i] = x, y, r
        changed = False
        for _ in range(3):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - 1e-12:
                        deficit = need - d
                        s = max((d - 1e-12) / (need + 1e-12), 0.0)
                        rad[i] = max(rad[i] * s, 0.0)
                        rad[j] = max(rad[j] * s, 0.0)
                        changed = True
        if not changed:
            break
    # final validity check
    r_final = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    feasible = True
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < r_final[i] + r_final[j] - 1e-9:
                feasible = False
                break
        if (pos[i, 0] < r_final[i] - 1e-9 or pos[i, 0] > 1.0 - r_final[i] + 1e-9 or
                pos[i, 1] < r_final[i] - 1e-9 or pos[i, 1] > 1.0 - r_final[i] + 1e-9):
            feasible = False
            break
    return pos, r_final, feasible

def _hex_seed(jitter=0.0, rng=None):
    """Hexagonal lattice seed."""
    rng = rng or np.random.default_rng()
    pts = []
    rows = 0
    while len(pts) < N:
        row = rows
        y = row * 0.8660254
        offset = (0.5 if row % 2 else 0.0)
        col = 0
        while len(pts) < N:
            x = col + offset
            pts.append((x, y))
            col += 1
        rows += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pts = (pts - [xmin, ymin]) / span
    cx = (xmax - xmin) / span
    cy = (ymax - ymin) / span
    pts[:, 0] = 0.5 - cx / 2 + pts[:, 0] * cx
    pts[:, 1] = 0.5 - cy / 2 + pts[:, 1] * cy
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    rad = np.full(N, 0.15)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad

def _local_search(pos, n_iter=100, step_size=0.004, seed=42, deadline=None):
    """Local search: joint position/radius optimization."""
    # We use L-BFGS-B on the soft objective with a low lambda to refine
    # This allows moving positions to allow larger radii
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_rad = _max_radius_for_positions(current_pos)
    current_obj = -np.sum(current_rad)
    
    z = np.empty(3 * N)
    z[0::3] = current_pos[:, 0]
    z[1::3] = current_pos[:, 1]
    z[2::3] = current_rad
    
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    
    # Try a few restarts with different lambdas
    best_z = z
    best_val = -current_obj
    
    for lam in [0.5, 0.1, 0.01]:
        if deadline is not None and time.time() > deadline:
            break
        try:
            res = minimize(_soft_objective, best_z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': 50, 'gtol': 1e-8, 'ftol': 1e-10})
            if time.time() > deadline:
                continue
            z_new = res.x
            p_new = np.column_stack([z_new[0::3], z_new[1::3]])
            r_new = z_new[2::3]
            p_new, r_new, feas = _project_feasible(p_new, r_new)
            if feas:
                val = float(np.sum(r_new))
                if val > best_val:
                    best_val = val
                    best_z = np.empty(3*N)
                    best_z[0::3] = p_new[:, 0]
                    best_z[1::3] = p_new[:, 1]
                    best_z[2::3] = r_new
        except Exception:
            pass
            
    p_final = np.column_stack([best_z[0::3], best_z[1::3]])
    r_final = _max_radius_for_positions(p_final)
    return p_final, float(np.sum(r_final))

def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(12345)

    best_val = -np.inf
    best_pos = None

    # Generate diverse seeds
    seeds = []
    for jitter in [0.0, 0.02, 0.05, 0.08, 0.12]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append((p, r))
    for _ in range(10):
        p = rng.uniform(0.15, 0.85, size=(N, 2))
        r = rng.uniform(0.05, 0.15, size=N)
        seeds.append((p, r))
        
    # Grid initialization
    grid_p = np.array([(i/5.0, j/5.0) for i in range(6) for j in range(6)])[:N]
    grid_r = np.full(N, 0.1)
    seeds.append((grid_p, grid_r))

    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N

    def _try_seed(pos, rad, lam, max_iter, deadline):
        nonlocal best_val, best_pos
        z = np.empty(3 * N)
        z[0::3] = pos[:, 0]
        z[1::3] = pos[:, 1]
        z[2::3] = rad
        try:
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
        except Exception:
            return
        if time.time() > deadline:
            return
        z = res.x
        p = np.column_stack([z[0::3], z[1::3]])
        r = z[2::3]
        p, r, feas = _project_feasible(p, r)
        if feas:
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_pos = p.copy()

    # Phase 1: Force-directed optimization with lambda annealing
    deadline = _START + _DEADLINE
    for si, (seed_pos, seed_rad) in enumerate(seeds):
        if time.time() > deadline - 20:
            break
        # Anneal lambda: start high to resolve overlaps, end low to maximize radii
        for lam in [50.0, 20.0, 10.0, 5.0, 2.0, 1.0]:
            if time.time() > deadline - 15:
                break
            _try_seed(seed_pos, seed_rad, lam, max_iter=100, deadline=deadline - 10)
            # Use current best as new seed for next lambda
            if best_pos is not None:
                seed_pos = best_pos
                seed_rad = _max_radius_for_positions(best_pos)

    if best_pos is None:
        pos = _hex_seed(jitter=0.0, rng=rng)[0]
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Local search refinement (Joint Optimization)
    if time.time() < deadline - 10:
        ls_pos, ls_obj = _local_search(best_pos, n_iter=100, step_size=0.003,
                                       seed=777, deadline=deadline - 8)
        if ls_obj > best_val:
            best_pos = ls_pos
            best_val = ls_obj

    # Phase 3: Final gradient polish
    if time.time() < deadline - 5:
        r_init = _max_radius_for_positions(best_pos)
        z = np.empty(3 * N)
        z[0::3] = best_pos[:, 0]
        z[1::3] = best_pos[:, 1]
        z[2::3] = r_init
        try:
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(0.1,),
                           options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            p, r, feas = _project_feasible(p, r)
            if feas:
                val = float(np.sum(r))
                if val > best_val:
                    best_pos = p
                    best_val = val
        except Exception:
            pass

    # Build result with computed radii
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
