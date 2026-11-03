"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(xyz, tol=1e-7):
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _enforce_validity(params):
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    for _ in range(500):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-10
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [500.0, 2000.0, 10000.0, 50000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 3000, 'ftol': 1e-15, 
                                      'gtol': 1e-13, 'maxfun': 15000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x.copy()
                best_val = obj_val
        except Exception:
            pass
    
    if time.time() - start_time < time_budget:
        penalty_weight = 100000.0
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight, smooth_eps=1e-5)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight, smooth_eps=1e-5)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 5000, 'ftol': 1e-16, 
                                      'gtol': 1e-14, 'maxfun': 20000})
            obj_val, _ = _objective_grad(result.x, penalty_weight, smooth_eps=1e-5)
            if obj_val < best_val:
                best_params = result.x.copy()
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def _mma_optimize(xyz, time_budget=5.0):
    """
    Method of Moving Asymptotes (MMA) optimization.
    Constructs convex quadratic subproblems to handle non-convex constraints.
    """
    start_time = time.time()
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    # MMA parameters
    gamma = 1.5  # exponent for asymptotes
    max_iter = 500
    tol = 1e-10
    
    # Bounds
    x_lo = np.zeros(N)
    x_hi = np.ones(N)
    y_lo = np.zeros(N)
    y_hi = np.ones(N)
    r_lo = np.full(N, 1e-6)
    r_hi = np.full(N, 0.5)
    
    # Initial reference points (for MMA asymptotes)
    x_ref = x.copy()
    y_ref = y.copy()
    r_ref = r.copy()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute objective and gradient
        f = -np.sum(r)
        df_dx = np.zeros(N)
        df_dy = np.zeros(N)
        df_dr = -np.ones(N)
        
        # Compute constraint gradients
        # Constraints: g_i(x) <= 0
        # 1. Boundary constraints: r_i - x_i <= 0, x_i + r_i - 1 <= 0, etc.
        # 2. Overlap constraints: r_i + r_j - dist_ij <= 0
        
        # For MMA, we need g(x) and dg/dx at current point
        # We'll use a penalty-based approach for the subproblem
        
        # Build the MMA subproblem
        # Minimize: f(x) + sum_i [gamma * (x_i / x_ref_i)^gamma + gamma * (x_ref_i / x_i)^(gamma-1) - (gamma-1) * (x_ref_i / x_i)^(gamma-1) * x_i/x_ref_i]
        # subject to linearized constraints
        
        # For simplicity, use a trust-region approach with MMA-like step control
        
        # Compute step direction using gradient
        # Project gradient to feasible region
        
        # Simple gradient step with adaptive step size
        step_size = 0.01
        
        # Compute gradient of objective + penalty
        penalty_weight = 1000.0
        params = np.concatenate([x, y, r])
        obj_val, grad = _objective_grad(params, penalty_weight, smooth_eps=1e-3)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Adaptive step size
        step_size = 0.01 * min(1.0, 0.1 / grad_norm)
        
        # Update
        new_params = params - step_size * grad
        
        new_x = new_params[0::3]
        new_y = new_params[1::3]
        new_r = new_params[2::3]
        
        # Enforce bounds
        new_x = np.clip(new_x, x_lo, x_hi)
        new_y = np.clip(new_y, y_lo, y_hi)
        new_r = np.clip(new_r, r_lo, r_hi)
        
        # Check if this improves the true objective
        new_score = np.sum(new_r)
        
        # Enforce validity by shrinking
        test_xyz = np.column_stack([new_x, new_y, new_r])
        test_xyz = _enforce_validity(test_xyz)
        valid_score = np.sum(test_xyz[:, 2])
        
        if valid_score > best_score:
            best_score = valid_score
            best_xyz = test_xyz.copy()
            x, y, r = test_xyz[:, 0], test_xyz[:, 1], test_xyz[:, 2]
            # Update reference points
            x_ref = x.copy()
            y_ref = y.copy()
            r_ref = r.copy()
        else:
            # Reduce step size
            step_size *= 0.5
    
    return best_xyz


def _physics_relax(xyz, time_limit=3.0, grow_rate=0.0015):
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.004
    velocity_x = np.zeros(N)
    velocity_y = np.zeros(N)
    damping = 0.7
    
    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break
        
        r *= (1.0 + grow_rate)
        
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    force = 100.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 100.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= 100.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += 100.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= 100.0 * (y[i] - (1 - r[i]))
        
        velocity_x = velocity_x * damping + dt * fx
        velocity_y = velocity_y * damping + dt * fy
        
        x += velocity_x
        y += velocity_y
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xyz = xyz.copy()
    
    return best_xyz


def _greedy_placement():
    xyz = np.zeros((N, 3))
    placed = np.zeros(N, dtype=bool)
    
    for idx in range(N):
        best_pos = (0.5, 0.5)
        best_r = 0.0
        
        for cx in np.linspace(0.02, 0.98, 20):
            for cy in np.linspace(0.02, 0.98, 20):
                r_wall = min(cx, 1 - cx, cy, 1 - cy)
                if r_wall <= 0.005:
                    continue
                
                min_dist = np.inf
                for j in range(idx):
                    if not placed[j]:
                        continue
                    dx = cx - xyz[j, 0]
                    dy = cy - xyz[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    max_r = max(0, d - xyz[j, 2])
                    min_dist = min(min_dist, max_r)
                
                r_candidate = min(r_wall, min_dist)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_pos = (cx, cy)
        
        xyz[idx, 0] = best_pos[0]
        xyz[idx, 1] = best_pos[1]
        xyz[idx, 2] = max(best_r, 0.01)
        placed[idx] = True
    
    return xyz


def _corner_anchored():
    xyz = np.zeros((N, 3))
    
    corner_r = 0.15
    xyz[0] = [corner_r, corner_r, corner_r]
    xyz[1] = [1 - corner_r, corner_r, corner_r]
    xyz[2] = [corner_r, 1 - corner_r, corner_r]
    xyz[3] = [1 - corner_r, 1 - corner_r, corner_r]
    
    edge_r = 0.10
    xyz[4] = [0.5, edge_r, edge_r]
    xyz[5] = [0.5, 1 - edge_r, edge_r]
    xyz[6] = [edge_r, 0.5, edge_r]
    xyz[7] = [1 - edge_r, 0.5, edge_r]
    
    remaining = N - 8
    interior_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.25 + i * 0.1667
            y = 0.25 + j * 0.1667
            interior_positions.append([x, y])
    
    interior_positions = interior_positions[:remaining]
    for i, pos in enumerate(interior_positions):
        xyz[8 + i] = [pos[0], pos[1], 0.06]
    
    return xyz


def _hex_initial():
    positions = []
    row_height = 0.155
    for row in range(7):
        y = 0.075 + row * row_height
        if row % 2 == 0:
            for col in range(5):
                x = 0.1 + col * 0.2
                positions.append([x, y])
        else:
            for col in range(5):
                x = 0.2 + col * 0.2
                positions.append([x, y])
        if len(positions) >= N:
            break
    positions = positions[:N]
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(N, dtype=np.float64) * 0.075
    return np.column_stack([positions, radii])


def _grid_initial():
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    positions.append([0.5, 0.5])
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(N, dtype=np.float64) * 0.08
    return np.column_stack([positions, radii])


def _random_initial(seed):
    np.random.seed(seed)
    positions = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.ones(N, dtype=np.float64) * 0.05
    return np.column_stack([positions, radii])


def construct():
    start_time = time.time()
    deadline = _TIME_LIMIT - 5.0
    
    best_result = None
    best_score = -np.inf
    
    configs = []
    configs.append(_corner_anchored())
    configs.append(_greedy_placement())
    configs.append(_hex_initial())
    configs.append(_grid_initial())
    for seed in [42, 123, 777, 2024]:
        configs.append(_random_initial(seed))
    
    num_configs = len(configs)
    time_per_config = (deadline - _elapsed()) / num_configs
    time_per_config = max(time_per_config, 2.0)
    
    for config_idx, xyz in enumerate(configs):
        if _elapsed() > deadline - 15.0:
            break
        
        config_time = min(time_per_config, deadline - _elapsed() - 15.0)
        if config_time < 1.0:
            break
        
        # Split time: 30% physics, 35% MMA, 35% polish
        physics_time = config_time * 0.30
        mma_time = config_time * 0.35
        polish_time = config_time * 0.35
        
        # Physics relaxation
        xyz_relaxed = _physics_relax(xyz, time_limit=physics_time)
        
        # MMA optimization
        xyz_mma = _mma_optimize(xyz_relaxed, time_budget=mma_time)
        
        # Polish with analytic gradient
        xyz_polished = _polish_analytic(xyz_mma, time_budget=polish_time)
        
        # Enforce validity
        xyz_final = _enforce_validity(xyz_polished)
        score_final = np.sum(xyz_final[:, 2])
        
        if score_final > best_score:
            best_score = score_final
            best_result = xyz_final.copy()
    
    # Final dedicated polish for the best candidate
    if best_result is not None and _elapsed() < deadline - 5.0:
        remaining_time = deadline - _elapsed() - 5.0
        if remaining_time > 3.0:
            xyz_final_polish = _polish_analytic(best_result, time_budget=remaining_time)
            xyz_final_polish = _enforce_validity(xyz_final_polish)
            score_final_polish = np.sum(xyz_final_polish[:, 2])
            if score_final_polish > best_score:
                best_score = score_final_polish
                best_result = xyz_final_polish
    
    if best_result is None:
        best_result = _enforce_validity(configs[0])
    
    for i in range(N):
        best_result[i, 0] = np.clip(best_result[i, 0], 0, 1)
        best_result[i, 1] = np.clip(best_result[i, 1], 0, 1)
        max_r = min(best_result[i, 0], best_result[i, 1], 
                   1 - best_result[i, 0], 1 - best_result[i, 1])
        best_result[i, 2] = min(best_result[i, 2], max_r)
        best_result[i, 2] = max(best_result[i, 2], 1e-6)
    
    return best_result
# EVOLVE-BLOCK-END
