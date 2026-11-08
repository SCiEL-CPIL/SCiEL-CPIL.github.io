import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _compute_radii(centers, W):
    """
    Given centers and W (H=2-W), compute max valid radii.
    Returns (sum_r, radii) or (0.0, None) if infeasible.
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 0.0, None
    
    n = len(centers)
    r = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        d_bound = min(xi, W - xi, yi, H - yi)
        d_min = d_bound
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < d_min:
                d_min = d
        r[i] = 0.5 * d_min
    
    # Iterative reduction to satisfy all pairwise constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                if r[i] + r[j] > d + 1e-12:
                    excess = r[i] + r[j] - d
                    if r[i] > 1e-8 and r[j] > 1e-8:
                        scale_i = r[i] / (r[i] + r[j])
                        scale_j = r[j] / (r[i] + r[j])
                        r[i] -= excess * scale_i
                        r[j] -= excess * scale_j
                        changed = True
                    elif r[i] > 1e-8:
                        r[i] -= excess
                        changed = True
                    else:
                        r[j] -= excess
                        changed = True
        if not changed:
            break
    
    # Final validation
    for i in range(n):
        if r[i] < 1e-10:
            return 0.0, None
        xi, yi = centers[i]
        if xi < r[i] - 1e-9 or xi > W - r[i] + 1e-9:
            return 0.0, None
        if yi < r[i] - 1e-9 or yi > H - r[i] + 1e-9:
            return 0.0, None
    
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if r[i] + r[j] > d + 1e-9:
                return 0.0, None
    
    return np.sum(r), r

def _softmin_beta(centers, W, beta=200.0):
    """
    Compute sum of soft-min radii using log-sum-exp trick.
    For each center i, r_i = -1/beta * log(sum_j exp(-beta * d_ij))
    where d_ij are distances to boundaries and other centers.
    """
    H = 2.0 - W
    n = len(centers)
    sum_r = 0.0
    
    for i in range(n):
        xi, yi = centers[i]
        # Distances to boundaries
        d_left = xi
        d_right = W - xi
        d_bottom = yi
        d_top = H - yi
        
        # Soft min of boundary distances
        # r_i = -1/beta * log(exp(-beta*d_left) + exp(-beta*d_right) + exp(-beta*d_bottom) + exp(-beta*d_top) + sum_j exp(-beta*d_ij))
        
        # Compute log-sum-exp for numerical stability
        # First, get all distances
        dists = [d_left, d_right, d_bottom, d_top]
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            dists.append(d)
        
        # Soft min: r = -1/beta * log(sum(exp(-beta*d)))
        # For numerical stability, factor out max
        min_d = min(dists)
        log_sum = 0.0
        for d in dists:
            log_sum += math.exp(-beta * (d - min_d))
        
        r_i = min_d - (1.0/beta) * math.log(log_sum)
        sum_r += r_i
    
    return sum_r

def _softmin_objective_and_grad(centers_flat, W, beta=200.0):
    """
    Objective and gradient for soft-min formulation.
    """
    H = 2.0 - W
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    obj = 0.0
    grad = np.zeros_like(centers_flat)
    
    for i in range(n):
        xi, yi = centers[i]
        d_left = xi
        d_right = W - xi
        d_bottom = yi
        d_top = H - yi
        
        # Collect all distances and their gradients
        dists = [(d_left, -1.0, 0.0), (d_right, 1.0, 0.0), 
                 (d_bottom, 0.0, -1.0), (d_top, 0.0, 1.0)]
        
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d > 1e-12:
                # Gradient of d wrt xi, yi: (dx/d, dy/d)
                # Gradient of d wrt xj, yj: (-dx/d, -dy/d)
                dists.append((d, dx/d, dy/d))
        
        # Soft min computation
        min_d = min(d[0] for d in dists)
        
        # Compute weights: w_k = exp(-beta*(d_k - min_d)) / sum(exp(-beta*(d_m - min_d)))
        exps = [math.exp(-beta * (d - min_d)) for d, _, _ in dists]
        sum_exps = sum(exps)
        
        # r_i = min_d - (1/beta)*log(sum_exps)
        r_i = min_d - (1.0/beta) * math.log(sum_exps)
        obj -= r_i
        
        # Gradient: dr_i/dx = sum_k w_k * dd_k/dx
        # where w_k = exp(-beta*(d_k-min_d)) / sum_exps
        # Note: min_d is piecewise constant, so we ignore its gradient contribution
        # (or more precisely, the gradient of the soft min is the weighted average of gradients)
        
        gx = 0.0
        gy = 0.0
        for k, (d, gd_dx, gd_dy) in enumerate(dists):
            w_k = exps[k] / sum_exps
            gx += w_k * gd_dx
            gy += w_k * gd_dy
        
        # Gradient of objective wrt center i
        grad[2*i] += -gx  # d(obj)/dx_i = -dr_i/dx_i
        grad[2*i+1] += -gy
        
        # Also affects other centers through their soft mins
        # This is handled when we process those centers
    
    return obj, grad

def _physics_relax(centers, W, H, dt=0.001, damping=0.95, max_steps=2000, deadline=None):
    """
    Physics-based relaxation using velocity-Verlet with repulsive forces.
    """
    n = len(centers)
    centers = centers.copy()
    velocities = np.zeros_like(centers)
    
    # Adaptive parameters
    force_strength = 10.0
    boundary_force = 50.0
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(centers)
        
        for i in range(n):
            xi, yi = centers[i]
            
            # Boundary repulsion
            margin = 0.05
            if xi < margin:
                forces[i, 0] += boundary_force * (margin - xi)
            elif xi > W - margin:
                forces[i, 0] -= boundary_force * (xi - (W - margin))
            if yi < margin:
                forces[i, 1] += boundary_force * (margin - yi)
            elif yi > H - margin:
                forces[i, 1] -= boundary_force * (yi - (H - margin))
            
            # Repulsion from other circles
            for j in range(n):
                if i == j:
                    continue
                dx = xi - centers[j][0]
                dy = yi - centers[j][1]
                d_sq = dx*dx + dy*dy
                d = math.sqrt(d_sq) if d_sq > 1e-12 else 1e-6
                
                # Inverse square repulsion (with cutoff)
                if d < 0.3:
                    f = force_strength / (d_sq + 1e-6)
                    forces[i, 0] += f * dx / d
                    forces[i, 1] += f * dy / d
        
        # Velocity Verlet integration
        velocities = velocities * damping + forces * dt
        centers += velocities * dt
        
        # Clamp to bounds
        centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
        centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
        
        # Check convergence
        if step > 100:
            vel_norm = np.linalg.norm(velocities)
            if vel_norm < 1e-6:
                break
    
    return centers

def _local_refine(centers, W, H, max_iter=50):
    """
    Local refinement: iteratively adjust each center to maximize its clearance.
    """
    centers = centers.copy()
    n = len(centers)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            best_pos = centers[i].copy()
            best_d = 0.0
            
            # Compute current min distance
            d_bound = min(centers[i][0], W - centers[i][0], centers[i][1], H - centers[i][1])
            d_min = d_bound
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < d_min:
                    d_min = d
            best_d = d_min
            
            # Try small perturbations
            for step in [0.005, 0.002, 0.001]:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = centers[i][0] + dx
                        ny = centers[i][1] + dy
                        if nx < 0.001 or nx > W - 0.001:
                            continue
                        if ny < 0.001 or ny > H - 0.001:
                            continue
                        
                        d_bound = min(nx, W - nx, ny, H - ny)
                        d_min = d_bound
                        for j in range(n):
                            if i == j:
                                continue
                            dxj = nx - centers[j][0]
                            dyj = ny - centers[j][1]
                            d = math.sqrt(dxj*dxj + dyj*dyj)
                            if d < d_min:
                                d_min = d
                        
                        if d_min > best_d + 1e-10:
                            best_d = d_min
                            best_pos = [nx, ny]
            
            if not np.allclose(best_pos, centers[i]):
                centers[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    return centers

def _grid_init(W, H, n_cols, n_rows, noise_scale=0.0):
    """Initialize centers in a grid pattern with optional noise."""
    centers = []
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            cx = (col + 0.5) * (W / n_cols)
            cy = (row + 0.5) * (H / n_rows)
            centers.append([cx, cy])
            idx += 1
        if idx >= N:
            break
    
    centers = np.array(centers)
    if noise_scale > 0:
        noise = np.random.uniform(-noise_scale, noise_scale, centers.shape)
        centers += noise
        centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
        centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    
    return centers

def _optimize_for_aspect(W, deadline):
    """
    Optimize for a given aspect ratio using multiple strategies.
    """
    H = 2.0 - W
    best_val = 0.0
    best_centers = None
    best_radii = None
    
    # Grid configurations
    if W < 0.9:
        configs = [(3, 7), (4, 6)]
    elif W > 1.1:
        configs = [(7, 3), (6, 4)]
    else:
        configs = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3)]
    
    for n_cols, n_rows in configs:
        if time.time() > deadline:
            break
        
        for noise_scale in [0.0, 0.02, 0.04]:
            if time.time() > deadline:
                break
            
            centers0 = _grid_init(W, H, n_cols, n_rows, noise_scale)
            
            # Strategy 1: Physics relaxation
            centers_physics = _physics_relax(centers0, W, H, max_steps=500, deadline=deadline)
            centers_refined = _local_refine(centers_physics, W, H, max_iter=30)
            val, radii = _compute_radii(centers_refined, W)
            
            if val > best_val:
                best_val = val
                best_centers = centers_refined.copy()
                best_radii = radii.copy()
            
            # Strategy 2: L-BFGS-B with soft-min objective
            if time.time() < deadline - 5:
                try:
                    r_min_est = 0.01
                    bounds = []
                    for i in range(N):
                        bounds.append((r_min_est, W - r_min_est))
                        bounds.append((r_min_est, H - r_min_est))
                    
                    res = minimize(
                        lambda v: _softmin_objective_and_grad(v, W, beta=200.0)[0],
                        lambda v: _softmin_objective_and_grad(v, W, beta=200.0)[1],
                        centers0.flatten(),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
                    )
                    
                    centers_opt = res.x.reshape(-1, 2)
                    centers_refined2 = _local_refine(centers_opt, W, H, max_iter=30)
                    val2, radii2 = _compute_radii(centers_refined2, W)
                    
                    if val2 > best_val:
                        best_val = val2
                        best_centers = centers_refined2.copy()
                        best_radii = radii2.copy()
                except Exception:
                    pass
    
    return best_centers, best_radii, W, best_val

def construct():
    deadline = time.time() + 140.0  # Safety margin (150s limit)
    
    best_val = 0.0
    best_centers = None
    best_radii = None
    best_W = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for W0 in aspect_ratios:
        if time.time() > deadline:
            break
        
        centers_opt, radii_opt, W_opt, val = _optimize_for_aspect(W0, deadline)
        
        if val > best_val:
            best_val = val
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
            best_W = W_opt
    
    # Final refinement on best aspect ratio
    if time.time() < deadline - 10 and best_centers is not None:
        W_best = best_W
        H_best = 2.0 - W_best
        
        # Multiple restarts with physics relaxation
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            centers0 = _grid_init(W_best, H_best, 5, 5, noise_scale=0.05)
            centers_physics = _physics_relax(centers0, W_best, H_best, max_steps=800, deadline=deadline)
            centers_refined = _local_refine(centers_physics, W_best, H_best, max_iter=40)
            val, radii = _compute_radii(centers_refined, W_best)
            
            if val > best_val:
                best_val = val
                best_centers = centers_refined.copy()
                best_radii = radii.copy()
    
    # Fallback: simple grid
    if best_val == 0.0 or best_centers is None:
        W = 1.0
        H = 1.0
        side = 5
        best_W = W
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            row = i // 5
            col = i % 5
            best_centers[i] = [(col + 0.5) / side, (row + 0.5) / side]
            best_radii[i] = 0.5 / side - 1e-4
    
    # Final validation
    val_final, radii_final = _compute_radii(best_centers, best_W)
    if val_final <= 0 or radii_final is None:
        W = 1.0
        H = 1.0
        side = 5
        best_W = W
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            row = i // 5
            col = i % 5
            best_centers[i] = [(col + 0.5) / side, (row + 0.5) / side]
            best_radii[i] = 0.5 / side - 1e-4
        radii_final = best_radii
    else:
        best_radii = radii_final
    
    result = np.column_stack([best_centers[:, 0], best_centers[:, 1], best_radii])
    return result

# EVOLVE-BLOCK-END
