"""Circle packing in perimeter-4 rectangle: contact-graph + SA + gradient optimization."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _compute_energy(params, W, H):
    """
    Compute objective: -sum(r) + penalties for overlaps and boundary violations.
    params: (3N,) array of [x1,y1,r1, x2,y2,r2, ...]
    """
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Sum of radii (we maximize this, so minimize negative)
    obj = -np.sum(r)
    
    # Penalty for negative radii
    neg_r = np.maximum(0, -r)
    obj += 1000.0 * np.sum(neg_r ** 2)
    
    # Penalty for boundary violations
    bl = np.maximum(0, -x + r)  # left
    br = np.maximum(0, x - W + r)  # right
    bb = np.maximum(0, -y + r)  # bottom
    bt = np.maximum(0, y - H + r)  # top
    obj += 1000.0 * (np.sum(bl ** 2) + np.sum(br ** 2) + np.sum(bb ** 2) + np.sum(bt ** 2))
    
    # Penalty for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                obj += 100.0 * overlap ** 2
    
    return obj


def _compute_gradient(params, W, H):
    """
    Compute gradient of the energy function.
    """
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] += -1.0  # d(-sum r)/dr_i = -1
    
    # Gradient of negative radius penalty
    for i in range(N):
        if r[i] < 0:
            grad[3*i+2] += 2000.0 * (-r[i])
    
    # Gradient of boundary penalties
    for i in range(N):
        # Left: max(0, -x_i + r_i)^2
        if -x[i] + r[i] > 0:
            grad[3*i+0] += 2000.0 * (-x[i] + r[i]) * (-1)  # d/dx of (-x+r)^2 = 2(-x+r)*(-1)
            grad[3*i+2] += 2000.0 * (-x[i] + r[i]) * (1)   # d/dr of (-x+r)^2 = 2(-x+r)*(1)
        # Right: max(0, x_i - W + r_i)^2
        if x[i] - W + r[i] > 0:
            grad[3*i+0] += 2000.0 * (x[i] - W + r[i]) * (1)
            grad[3*i+2] += 2000.0 * (x[i] - W + r[i]) * (1)
        # Bottom: max(0, -y_i + r_i)^2
        if -y[i] + r[i] > 0:
            grad[3*i+1] += 2000.0 * (-y[i] + r[i]) * (-1)
            grad[3*i+2] += 2000.0 * (-y[i] + r[i]) * (1)
        # Top: max(0, y_i - H + r_i)^2
        if y[i] - H + r[i] > 0:
            grad[3*i+1] += 2000.0 * (y[i] - H + r[i]) * (1)
            grad[3*i+2] += 2000.0 * (y[i] - H + r[i]) * (1)
    
    # Gradient of overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # d(overlap^2)/dx_i = 2*overlap * d(overlap)/dx_i
                # d(overlap)/dx_i = d(r_i + r_j - dist)/dx_i = -d(dist)/dx_i = -dx/dist
                # d(dist)/dx_i = dx/dist, so d(overlap)/dx_i = -dx/dist
                # d(overlap)/dx_j = dx/dist
                # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
                factor = 200.0 * overlap
                if dist > 1e-12:
                    grad[3*i+0] += factor * (-dx / dist)
                    grad[3*i+1] += factor * (-dy / dist)
                    grad[3*j+0] += factor * (dx / dist)
                    grad[3*j+1] += factor * (dy / dist)
                grad[3*i+2] += factor * 1.0
                grad[3*j+2] += factor * 1.0
    
    return grad


def _local_optimize(params, W, H, max_iter=2000):
    """
    Local optimization using L-BFGS-B with analytic gradients.
    """
    # Set bounds: r >= 0, x in [0, W], y in [0, H]
    bounds = []
    for i in range(N):
        bounds.append((0, W))      # x
        bounds.append((0, H))      # y
        bounds.append((0, min(W, H)/2))  # r
    
    result = minimize(
        _compute_energy,
        params,
        jac=_compute_gradient,
        args=(W, H),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x


def _simulated_annealing(params, W, H, start_temp=0.05, end_temp=0.001, 
                         n_steps=3000, step_scale=0.01):
    """
    Simulated annealing to escape local minima.
    """
    current = params.copy()
    current_energy = _compute_energy(current, W, H)
    best = current.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(12345)
    n_steps_actual = n_steps
    for step in range(n_steps_actual):
        # Cooling schedule (geometric)
        t = start_temp * (end_temp / start_temp) ** (step / max(n_steps_actual - 1, 1))
        
        # Generate perturbation
        scale = step_scale * (1.0 + 0.5 * np.sin(step * 0.1))
        delta = rng.normal(0, scale, size=current.shape)
        # Don't perturb radii as much
        delta[2::3] *= 0.3
        
        proposed = current + delta
        
        # Clip to bounds
        for i in range(N):
            proposed[3*i+0] = np.clip(proposed[3*i+0], 0, W)
            proposed[3*i+1] = np.clip(proposed[3*i+1], 0, H)
            proposed[3*i+2] = np.clip(proposed[3*i+2], 0, min(W, H)/2)
        
        proposed_energy = _compute_energy(proposed, W, H)
        delta_e = proposed_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / t)
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
            current = proposed
            current_energy = proposed_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
    
    return best, best_energy


def _optimize_for_aspect(ar, time_budget):
    """
    Optimize circle packing for a given aspect ratio.
    """
    W = 2.0 * ar / (ar + 1.0)
    H = 2.0 / (ar + 1.0)
    
    start = time.time()
    best_params = None
    best_energy = np.inf
    
    # Initialize with grid
    side = int(np.ceil(np.sqrt(N)))
    x_grid = np.array([(i + 0.5) * W / side for i in range(side)])
    y_grid = np.array([(j + 0.5) * H / side for j in range(side)])
    
    init_params = np.zeros(3 * N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            init_params[3*idx+0] = x_grid[i]
            init_params[3*idx+1] = y_grid[j]
            init_params[3*idx+2] = min(W, H) / (2 * side) * 0.9
            idx += 1
    
    # Local optimization from grid init
    local_result = _local_optimize(init_params, W, H, max_iter=1000)
    local_energy = _compute_energy(local_result, W, H)
    if local_energy < best_energy:
        best_energy = local_energy
        best_params = local_result.copy()
    
    elapsed = time.time() - start
    if elapsed < time_budget * 0.4:
        # Simulated annealing
        sa_params, sa_energy = _simulated_annealing(
            init_params, W, H,
            start_temp=0.03, end_temp=0.0005,
            n_steps=2000, step_scale=0.008
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_params = sa_params.copy()
        
        # Local optimization from SA result
        local2 = _local_optimize(sa_params, W, H, max_iter=1000)
        local2_energy = _compute_energy(local2, W, H)
        if local2_energy < best_energy:
            best_energy = local2_energy
            best_params = local2.copy()
    
    elapsed = time.time() - start
    if elapsed < time_budget * 0.7:
        # Final polish with more iterations
        polished = _local_optimize(best_params, W, H, max_iter=3000)
        polished_energy = _compute_energy(polished, W, H)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_params = polished.copy()
    
    # Extract circles
    circles = best_params.reshape(N, 3)
    # Ensure validity
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], W - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], H - circles[:, 2])
    circles[:, 2] = np.clip(circles[:, 2], 0.001, min(W, H) / 2)
    
    if not _check_validity(circles, W, H):
        return None, None
    
    return circles, W, H


def _optimize_perimeter(circles, W, H, time_budget):
    """
    Optimize the perimeter (W, H) while keeping 2(W+H) = 4.
    We parameterize by W, with H = 2 - W.
    """
    start = time.time()
    best_circles = circles.copy()
    best_W = W
    best_H = H
    best_sum = np.sum(circles[:, 2])
    
    # Try different W values
    w_candidates = np.linspace(0.5, 1.5, 11)  # W from 0.5 to 1.5, H = 2-W
    
    for w_new in w_candidates:
        h_new = 2.0 - w_new
        if h_new <= 0 or w_new <= 0:
            continue
        
        elapsed = time.time() - start
        if elapsed > time_budget - 5:
            break
        
        # Rescale circles to fit new rectangle
        scale_x = w_new / W
        scale_y = h_new / H
        new_circles = best_circles.copy()
        new_circles[:, 0] *= scale_x
        new_circles[:, 1] *= scale_y
        # Scale radii by min scale to maintain aspect
        scale_r = min(scale_x, scale_y)
        new_circles[:, 2] *= scale_r
        
        # Local optimization
        params = new_circles.flatten()
        optimized = _local_optimize(params, w_new, h_new, max_iter=1500)
        opt_circles = optimized.reshape(N, 3)
        opt_circles[:, 0] = np.clip(opt_circles[:, 0], opt_circles[:, 2], w_new - opt_circles[:, 2])
        opt_circles[:, 1] = np.clip(opt_circles[:, 1], opt_circles[:, 2], h_new - opt_circles[:, 2])
        opt_circles[:, 2] = np.clip(opt_circles[:, 2], 0.001, min(w_new, h_new) / 2)
        
        if _check_validity(opt_circles, w_new, h_new):
            s = np.sum(opt_circles[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = opt_circles.copy()
                best_W = w_new
                best_H = h_new
    
    return best_circles, best_W, best_H


def construct():
    start_time = time.time()
    best_circles = None
    best_sum = -np.inf
    best_W, best_H = 1.0, 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Allocate time based on aspect ratio
        if 0.8 <= ar <= 1.2:
            time_budget = 10.0
        elif 0.6 <= ar <= 1.5:
            time_budget = 8.0
        else:
            time_budget = 5.0
        
        circles, W, H = _optimize_for_aspect(ar, time_budget)
        if circles is not None:
            s = np.sum(circles[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()
                best_W, best_H = W, H
    
    # Perimeter optimization on the best result
    if best_circles is not None and time.time() - start_time < DEADLINE - 15:
        best_circles, best_W, best_H = _optimize_perimeter(
            best_circles, best_W, best_H,
            time_budget=min(20.0, DEADLINE - 15 - (time.time() - start_time))
        )
        best_sum = np.sum(best_circles[:, 2])
    
    # Final local optimization
    if best_circles is not None and time.time() - start_time < DEADLINE - 5:
        params = best_circles.flatten()
        polished = _local_optimize(params, best_W, best_H, max_iter=5000)
        polished_circles = polished.reshape(N, 3)
        polished_circles[:, 0] = np.clip(polished_circles[:, 0], polished_circles[:, 2], best_W - polished_circles[:, 2])
        polished_circles[:, 1] = np.clip(polished_circles[:, 1], polished_circles[:, 2], best_H - polished_circles[:, 2])
        polished_circles[:, 2] = np.clip(polished_circles[:, 2], 0.001, min(best_W, best_H) / 2)
        
        if _check_validity(polished_circles, best_W, best_H):
            s = np.sum(polished_circles[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = polished_circles.copy()
    
    # Fallback: simple grid in square
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
    
    # Final validity check
    if not _check_validity(best_circles, best_W, best_H):
        # Fallback to safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
