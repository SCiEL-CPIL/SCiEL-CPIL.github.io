"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _hexagonal_init():
    """Generate hexagonal lattice points and select N closest to center."""
    i_max = 20
    j_max = 20
    points = []
    for j in range(-j_max, j_max + 1):
        for i in range(-i_max, i_max + 1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    selected = points[idx]
    max_extent = max(np.max(np.abs(selected[:, 0])), np.max(np.abs(selected[:, 1])))
    s = 0.5 / (max_extent + 0.5)
    return selected * s

def _grid_init():
    """Generate a grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    margin = 0.1
    xs = np.linspace(-0.5 + margin, 0.5 - margin, side)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, side)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    # Shuffle and take N
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(points))[:N]
    return points[perm]

def _random_init(seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _soft_min_obj(x, beta=50.0):
    """
    Smooth surrogate for the max-min problem using log-sum-exp.
    We maximize min(d_pair/2, d_wall) by minimizing a smooth function.
    
    Uses auxiliary variable t (flattened into x) with epigraph formulation:
    x = [centers (2N), t]
    Minimize: -t + penalty for constraints d_ij >= 2t and wall_i >= t
    """
    centers = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Soft constraint: sum of soft violations
    # For pairs: want dists >= 2t, so violation = max(0, 2t - dists)
    # For walls: want wall_dists >= t, so violation = max(0, t - wall_dists)
    
    pair_viol = np.maximum(0, 2*t - dists)
    wall_viol = np.maximum(0, t - wall_dists)
    
    # Use soft-max (log-sum-exp) for smoothness
    # But first, let's use a simpler smooth approximation with exponentials
    # violation_soft = sum(exp(beta * violation) - 1) / beta
    
    obj = -t
    obj += np.sum(np.exp(beta * pair_viol) - 1.0) / beta
    obj += np.sum(np.exp(beta * wall_viol) - 1.0) / beta
    
    return obj

def _soft_min_grad(x, beta=50.0):
    """Gradient of the soft-min objective."""
    centers = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    pair_viol = np.maximum(0, 2*t - dists)
    wall_viol = np.maximum(0, t - wall_dists)
    
    # d(obj)/dt = -1 + sum of gradients of soft violations
    grad_t = -1.0
    # d/dt of exp(beta*(2t-d))/beta = 2*exp(beta*(2t-d)) where 2t>d
    mask_pair = (2*t > dists) & (np.eye(N, dtype=bool) == False)
    if np.any(mask_pair):
        grad_t += 2.0 * np.sum(np.exp(beta * pair_viol[mask_pair]))
    mask_wall = (t > wall_dists)
    if np.any(mask_wall):
        grad_t += np.sum(np.exp(beta * wall_viol[mask_wall]))
    
    # d(obj)/d(centers[i]) from pair violations
    grad_c = np.zeros_like(centers)
    for i in range(N):
        for j in range(i+1, N):
            d = dists[i, j]
            if d < 2*t and d > 1e-12:
                # d/d(centers[i]) of exp(beta*(2t-d))/beta
                # = exp(beta*(2t-d)) * (-beta) * d/d(centers[i]) d
                # d/d(centers[i]) d = (centers[i] - centers[j]) / d
                exp_term = np.exp(beta * (2*t - d))
                v = (centers[i] - centers[j]) / d
                grad_c[i] -= exp_term * v
                grad_c[j] += exp_term * v
    
    # d(obj)/d(centers[i]) from wall violations
    for i in range(N):
        if wall_dists[i] < t:
            exp_term = np.exp(beta * (t - wall_dists[i]))
            x_i, y_i = centers[i]
            # wall_dist = min(0.5-|x|, 0.5-|y|)
            if 0.5 - abs(x_i) <= 0.5 - abs(y_i):
                # x is closer to wall
                if x_i > 0:
                    grad_c[i, 0] += exp_term
                else:
                    grad_c[i, 0] -= exp_term
            else:
                # y is closer to wall
                if y_i > 0:
                    grad_c[i, 1] += exp_term
                else:
                    grad_c[i, 1] -= exp_term
    
    grad = np.zeros_like(x)
    grad[:2*N] = grad_c.ravel()
    grad[2*N] = grad_t
    
    return grad

def _l_bfgs_b_optimization(centers, start_time, timeout, beta=50.0):
    """L-BFGS-B optimization with soft-min objective."""
    x0 = np.concatenate([centers.ravel(), [0.03]])
    
    # Bounds for centers
    bounds_c = [(-0.5, 0.5)] * (2*N)
    bounds_t = [(0.0, 0.5)]
    bounds = bounds_c + bounds_t
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    def callback(xk):
        nonlocal best_r, best_centers
        ck = xk[:2*N].reshape(N, 2)
        r = _compute_radius(ck)
        if r > best_r:
            best_r = r
            best_centers = ck.copy()
        return False
    
    try:
        result = minimize(
            _soft_min_obj, x0, args=(beta,),
            jac=_soft_min_grad, method='L-BFGS-B',
            bounds=bounds, callback=callback,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception:
        pass
    
    # Also try with the result
    if hasattr(result, 'x') and result.x is not None:
        ck = result.x[:2*N].reshape(N, 2)
        r = _compute_radius(ck)
        if r > best_r:
            best_r = r
            best_centers = ck.copy()
    
    return best_centers, best_r

def _force_relaxation(centers, start_time, timeout):
    """Force-based relaxation to maximize minimum distance."""
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    current_r = best_r
    
    step = 0.01
    stall = 0
    
    while time.time() - start_time < timeout:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        min_pair = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        current_r = min(min_pair / 2.0, min_wall)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            stall = 0
        else:
            stall += 1
        
        if stall > 200:
            break
        
        threshold = min_pair * 1.5
        forces = np.zeros_like(centers)
        
        mask = (dists < threshold) & (dists > 1e-10)
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = dists[i, j]
                    f = 1.0 / (d * d)
                    v = diff[i, j] / d
                    forces[i] += f * v
                    forces[j] -= f * v
        
        for i in range(N):
            x, y = centers[i]
            dx_l = x + 0.5
            dx_r = 0.5 - x
            dy_b = y + 0.5
            dy_t = 0.5 - y
            
            if dx_l < 0.1:
                forces[i, 0] += 1.0 / (dx_l * dx_l + 1e-10)
            if dx_r < 0.1:
                forces[i, 0] -= 1.0 / (dx_r * dx_r + 1e-10)
            if dy_b < 0.1:
                forces[i, 1] += 1.0 / (dy_b * dy_b + 1e-10)
            if dy_t < 0.1:
                forces[i, 1] -= 1.0 / (dy_t * dy_t + 1e-10)
        
        force_norms = np.linalg.norm(forces, axis=1)
        max_fn = np.max(force_norms)
        if max_fn > 1e-10:
            forces = forces / max_fn
        
        centers += step * forces
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        new_r = _compute_radius(centers)
        if new_r > current_r:
            step = min(step * 1.1, 0.05)
        else:
            step *= 0.9
            if step < 1e-5:
                break
    
    return best_centers, best_r

def _polish(centers, start_time, timeout):
    """High-precision polish: iteratively fix the most violated constraint."""
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = 1e-5
    stall = 0
    
    while time.time() - start_time < timeout:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        current_r = min(min_pair / 2.0, min_wall)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            stall = 0
        else:
            stall += 1
        
        if stall > 500:
            break
        
        if min_pair / 2.0 < min_wall:
            i, j = min_pair_idx
            v = centers[j] - centers[i]
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                v = np.array([1.0, 0.0])
            else:
                v = v / norm
            
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            
            centers[i] -= step * v
            centers[j] += step * v
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            centers[j] = np.clip(centers[j], -0.5, 0.5)
            
            new_r = _compute_radius(centers)
            if new_r <= current_r:
                centers[i] = old_i
                centers[j] = old_j
        else:
            i = min_wall_idx
            x, y = centers[i]
            if abs(x) >= abs(y):
                v = np.array([-np.sign(x), 0.0]) if abs(x) > 1e-12 else np.array([1.0, 0.0])
            else:
                v = np.array([0.0, -np.sign(y)]) if abs(y) > 1e-12 else np.array([0.0, 1.0])
            
            old_i = centers[i].copy()
            centers[i] += step * v
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            
            new_r = _compute_radius(centers)
            if new_r <= current_r:
                centers[i] = old_i
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    timeout = 140.0  # 10s margin for safety
    
    # Phase 1: Multiple initializations, pick best
    inits = [
        _hexagonal_init(),
        _grid_init(),
        _random_init(seed=42),
        _random_init(seed=123),
    ]
    
    best_centers = None
    best_r = 0.0
    for init in inits:
        r = _compute_radius(init)
        if r > best_r:
            best_r = r
            best_centers = init.copy()
    
    centers = best_centers
    
    # Phase 2: Force-based relaxation
    remaining = timeout - (time.time() - start_time)
    if remaining > 30.0:
        centers, r1 = _force_relaxation(centers, start_time, time.time() + remaining - 30.0)
    
    # Phase 3: L-BFGS-B optimization with soft-min
    remaining = timeout - (time.time() - start_time)
    if remaining > 20.0:
        centers, r2 = _l_bfgs_b_optimization(centers, start_time, time.time() + remaining - 20.0)
    
    # Phase 4: High-precision polish
    remaining = timeout - (time.time() - start_time)
    if remaining > 5.0:
        centers, r3 = _polish(centers, start_time, time.time() + remaining)
    
    # Final validation
    final_r = _compute_radius(centers)
    if final_r <= 0 or not np.isfinite(final_r):
        centers = _hexagonal_init()
    
    return centers
# EVOLVE-BLOCK-END
