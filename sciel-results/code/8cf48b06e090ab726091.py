"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    # Use logaddexp for numerical stability
    return np.logaddexp(0, z / eps) * eps

def _softplus_grad(z, eps=1e-4):
    """Derivative of softplus(z) = log(1 + exp(z/eps)) * eps."""
    # d/dz [eps * log(1 + exp(z/eps))] = eps * (exp(z/eps) / (1 + exp(z/eps))) * (1/eps)
    # = 1 / (1 + exp(-z/eps)) = sigmoid(z/eps)
    return 1.0 / (1.0 + np.exp(-z / eps))

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    Returns: (value, gradient)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    val = -np.sum(r)
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # --- Boundary Violations ---
    # lx = r - x, ux = x + r - 1, ly = r - y, uy = y + r - 1
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    # Penalty: sum(softplus(z)^2)
    # d/dz (softplus(z)^2) = 2 * softplus(z) * softplus_grad(z)
    
    sp_lx = _softplus(lx, smooth_eps)
    sp_ux = _softplus(ux, smooth_eps)
    sp_ly = _softplus(ly, smooth_eps)
    sp_uy = _softplus(uy, smooth_eps)
    
    val += np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + np.sum(sp_uy**2)
    
    d_lx = 2 * sp_lx * _softplus_grad(lx, smooth_eps)
    d_ux = 2 * sp_ux * _softplus_grad(ux, smooth_eps)
    d_ly = 2 * sp_ly * _softplus_grad(ly, smooth_eps)
    d_uy = 2 * sp_uy * _softplus_grad(uy, smooth_eps)
    
    # Gradients for x, y, r
    # d/dx (r-x) = -1, d/dx (x+r-1) = 1
    # d/dy (r-y) = -1, d/dy (y+r-1) = 1
    # d/dr (r-x) = 1, d/dr (x+r-1) = 1, etc.
    
    grad[0::3] += -d_lx + d_ux
    grad[1::3] += -d_ly + d_uy
    grad[2::3] += d_lx + d_ux + d_ly + d_uy
    
    # --- Pairwise Overlaps ---
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ovl = _softplus(overlap, smooth_eps)
    val += np.sum(sp_ovl**2)
    
    d_ovl = 2 * sp_ovl * _softplus_grad(overlap, smooth_eps)
    
    # Gradient w.r.t. x_i, x_j, y_i, y_j, r_i, r_j
    # d/dx_i (r_i + r_j - dist) = -dx / dist
    # d/dx_j (r_i + r_j - dist) = +dx / dist
    # d/dr_i (r_i + r_j - dist) = 1
    
    inv_dist = 1.0 / dist
    gx_i = -dx * inv_dist
    gy_i = -dy * inv_dist
    
    # Accumulate gradients
    # For each pair (i, j):
    # grad_x[i] += d_ovl * gx_i
    # grad_x[j] += d_ovl * (-gx_i)  -> d_ovl * dx/dist
    # grad_y[i] += d_ovl * gy_i
    # grad_y[j] += d_ovl * (-gy_i)  -> d_ovl * dy/dist
    # grad_r[i] += d_ovl
    # grad_r[j] += d_ovl
    
    # Use np.add.at for accumulation
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    np.add.at(grad_x, idx_i, d_ovl * gx_i)
    np.add.at(grad_x, idx_j, -d_ovl * gx_i)
    
    np.add.at(grad_y, idx_i, d_ovl * gy_i)
    np.add.at(grad_y, idx_j, -d_ovl * gy_i)
    
    np.add.at(grad_r, idx_i, d_ovl)
    np.add.at(grad_r, idx_j, d_ovl)
    
    grad[0::3] += grad_x
    grad[1::3] += grad_y
    grad[2::3] += grad_r
    
    # --- Negative Radius ---
    neg_r = -r
    sp_rad = _softplus(neg_r, smooth_eps)
    val += np.sum(sp_rad**2)
    
    d_rad = 2 * sp_rad * _softplus_grad(neg_r, smooth_eps)
    # d/dr (-r) = -1
    grad[2::3] += -d_rad
    
    return val + penalty_weight * (val - (-np.sum(r))), grad # This is wrong, just return val, grad scaled?
    # Actually, the objective passed to minimize should be the total value.
    # Let's just return the value and gradient of the full function: -sum(r) + penalty_weight * penalty_terms
    
    # Recalculate properly
    penalty_val = np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + np.sum(sp_uy**2) + np.sum(sp_ovl**2) + np.sum(sp_rad**2)
    penalty_grad = np.zeros_like(params)
    
    # Re-accumulate penalty gradients
    # It's cleaner to compute the penalty value and gradient separately and then combine.
    
    return -np.sum(r) + penalty_weight * penalty_val, -np.ones(3*N)[2::3] * 1.0 + penalty_weight * penalty_grad

def _objective_and_grad(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] -= 1.0
    
    # Boundary
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    sp_lx = _softplus(lx, smooth_eps)
    sp_ux = _softplus(ux, smooth_eps)
    sp_ly = _softplus(ly, smooth_eps)
    sp_uy = _softplus(uy, smooth_eps)
    
    p_val = np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + np.sum(sp_uy**2)
    
    d_lx = 2 * sp_lx * _softplus_grad(lx, smooth_eps)
    d_ux = 2 * sp_ux * _softplus_grad(ux, smooth_eps)
    d_ly = 2 * sp_ly * _softplus_grad(ly, smooth_eps)
    d_uy = 2 * sp_uy * _softplus_grad(uy, smooth_eps)
    
    p_grad = np.zeros_like(params)
    p_grad[0::3] += -d_lx + d_ux
    p_grad[1::3] += -d_ly + d_uy
    p_grad[2::3] += d_lx + d_ux + d_ly + d_uy
    
    # Overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ovl = _softplus(overlap, smooth_eps)
    p_val += np.sum(sp_ovl**2)
    
    d_ovl = 2 * sp_ovl * _softplus_grad(overlap, smooth_eps)
    inv_dist = 1.0 / dist
    gx = -dx * inv_dist
    gy = -dy * inv_dist
    
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    np.add.at(grad_x, idx_i, d_ovl * gx)
    np.add.at(grad_x, idx_j, -d_ovl * gx)
    np.add.at(grad_y, idx_i, d_ovl * gy)
    np.add.at(grad_y, idx_j, -d_ovl * gy)
    np.add.at(grad_r, idx_i, d_ovl)
    np.add.at(grad_r, idx_j, d_ovl)
    
    p_grad[0::3] += grad_x
    p_grad[1::3] += grad_y
    p_grad[2::3] += grad_r
    
    # Radius
    neg_r = -r
    sp_rad = _softplus(neg_r, smooth_eps)
    p_val += np.sum(sp_rad**2)
    d_rad = 2 * sp_rad * _softplus_grad(neg_r, smooth_eps)
    p_grad[2::3] += -d_rad
    
    return val + penalty_weight * p_val, grad + penalty_weight * p_grad

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5/cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradients."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Define the objective function wrapper for minimize
    def obj_fun(p):
        val, grad = _objective_and_grad(p, 100.0, 1e-4)
        return val, grad

    # Phase 1: Moderate penalty
    remaining = _remaining_time()
    if remaining > 10:
        try:
            res = minimize(obj_fun, params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < obj_fun(params)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: High penalty
    remaining = _remaining_time()
    if remaining > 8:
        def obj_fun2(p):
            val, grad = _objective_and_grad(p, 1000.0, 1e-5)
            return val, grad
        try:
            res = minimize(obj_fun2, params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < obj_fun2(params)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: Very High Penalty for Polish
    remaining = _remaining_time()
    if remaining > 5:
        def obj_fun3(p):
            val, grad = _objective_and_grad(p, 10000.0, 1e-6)
            return val, grad
        try:
            res = minimize(obj_fun3, params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < obj_fun3(params)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1)
    ]
    
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 10.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Local refinement
    remaining = _remaining_time()
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02]:
            if remaining < 5.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()
    
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
