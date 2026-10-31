"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-4)
    
    penalty = 0.0
    # Boundary penalties (smooth)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * min_dist_sq[mask]))
    
    return -np.sum(radii) + 500.0 * penalty


def _objective_grad(x):
    """Objective with analytical gradient for faster L-BFGS-B convergence."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    r_safe = np.maximum(radii, 1e-4)
    
    grad = np.zeros_like(x)
    
    # Gradient of -sum(radii)
    grad[2*n:] = -1.0
    
    # Boundary gradients
    # left: max(0, r - x)^2 -> 2*(r-x) if r > x else 0
    left_viol = np.maximum(0.0, r_safe - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - r_safe))
    bottom_viol = np.maximum(0.0, r_safe - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - r_safe))
    
    # d/dx of (r - x)^2 = -2(r - x)
    grad[:n] += -2.0 * left_viol
    # d/dx of (x - (1-r))^2 = 2(x - (1-r))
    grad[:n] += 2.0 * right_viol
    # d/dy of (r - y)^2 = -2(r - y)
    grad[n:2*n] += -2.0 * bottom_viol
    # d/dy of (y - (1-r))^2 = 2(y - (1-r))
    grad[n:2*n] += 2.0 * top_viol
    
    # d/dr of (r - x)^2 = 2(r - x)
    grad[2*n:] += 2.0 * left_viol
    # d/dr of (x - (1-r))^2 = -2(x - (1-r))
    grad[2*n:] += -2.0 * right_viol
    # d/dr of (r - y)^2 = 2(r - y)
    grad[2*n:] += 2.0 * bottom_viol
    # d/dr of (y - (1-r))^2 = -2(y - (1-r))
    grad[2*n:] += -2.0 * top_viol
    
    # Pairwise overlap gradients
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = r_safe[:, np.newaxis] + r_safe[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # overlap = max(0, min_dist_sq - dist_sq)
    # penalty term = overlap^2 / (4 * min_dist_sq)
    # d/d(overlap) = 2*overlap / (4*min_dist_sq) = overlap / (2*min_dist_sq)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    
    # Only compute gradients where overlap > 0
    active = mask & (overlap > 0)
    
    # d(overlap)/d(dist_sq) = -1
    # d(overlap)/d(x_i) = -2*(x_i - x_j) = -2*dx[i,j]
    # d(overlap)/d(y_i) = -2*(y_i - y_j) = -2*dy[i,j]
    # d(overlap)/d(r_i) = 2*min_dist[i,j]
    # d(overlap)/d(r_j) = 2*min_dist[i,j]
    
    # For each active pair (i,j), i < j:
    # dP/dx_i += (overlap/(2*min_dist_sq)) * (-2*dx[i,j])
    # dP/dx_j += (overlap/(2*min_dist_sq)) * (2*dx[i,j])
    # dP/dy_i += (overlap/(2*min_dist_sq)) * (-2*dy[i,j])
    # dP/dy_j += (overlap/(2*min_dist_sq)) * (2*dy[i,j])
    # dP/dr_i += (overlap/(2*min_dist_sq)) * 2*min_dist[i,j]
    # dP/dr_j += (overlap/(2*min_dist_sq)) * 2*min_dist[i,j]
    
    # Vectorized approach using scatter-add
    coeff = np.zeros((n, n))
    coeff[active] = overlap[active] / (2.0 * min_dist_sq[active])
    
    # Make symmetric for gradient computation
    coeff_sym = coeff + coeff.T
    
    # dx and dy are antisymmetric: dx[j,i] = -dx[i,j]
    # dP/dx_i = sum_j coeff_sym[i,j] * (-2*dx[i,j]) ... wait, need to be careful
    
    # Actually let's just do it more directly
    # For each active pair (i,j):
    # grad_x[i] += coeff[i,j] * (-2*dx[i,j])
    # grad_x[j] += coeff[i,j] * (2*dx[i,j])
    # grad_y[i] += coeff[i,j] * (-2*dy[i,j])
    # grad_y[j] += coeff[i,j] * (2*dy[i,j])
    # grad_r[i] += coeff[i,j] * 2*min_dist[i,j]
    # grad_r[j] += coeff[i,j] * 2*min_dist[i,j]
    
    # Using matrix operations:
    # grad_x[i] = -2 * sum_j coeff[i,j] * dx[i,j]
    # But coeff is only upper triangle, so we need to handle symmetry
    
    # Let's use the symmetric version
    # grad_x[i] = -2 * sum_j (coeff[i,j] + coeff[j,i]) * dx[i,j] / 2 ... no
    
    # Simpler: just loop over active pairs since n=26 is small
    for i in range(n):
        for j in range(i+1, n):
            if active[i, j]:
                c = coeff[i, j]
                dx_ij = dx[i, j]
                dy_ij = dy[i, j]
                md = min_dist[i, j]
                
                grad[i] += c * (-2.0 * dx_ij)
                grad[j] += c * (2.0 * dx_ij)
                grad[n+i] += c * (-2.0 * dy_ij)
                grad[n+j] += c * (2.0 * dy_ij)
                grad[2*n+i] += c * 2.0 * md
                grad[2*n+j] += c * 2.0 * md
    
    # Scale by penalty factor 500
    grad = grad * 1.0  # The -1 for radii is already set, penalty gradients need *500
    
    # Actually, let me recompute: objective = -sum(r) + 500 * penalty
    # So grad = [-1 for radii part] + 500 * [penalty gradients]
    
    # Let me redo this properly
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0  # from -sum(radii)
    
    # Penalty gradients * 500
    penalty_grad_x = np.zeros(n)
    penalty_grad_y = np.zeros(n)
    penalty_grad_r = np.zeros(n)
    
    # Boundary
    penalty_grad_x -= 2.0 * left_viol
    penalty_grad_x += 2.0 * right_viol
    penalty_grad_y -= 2.0 * bottom_viol
    penalty_grad_y += 2.0 * top_viol
    penalty_grad_r += 2.0 * left_viol
    penalty_grad_r -= 2.0 * right_viol
    penalty_grad_r += 2.0 * bottom_viol
    penalty_grad_r -= 2.0 * top_viol
    
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            if active[i, j]:
                c = coeff[i, j]
                dx_ij = dx[i, j]
                dy_ij = dy[i, j]
                md = min_dist[i, j]
                
                penalty_grad_x[i] += c * (-2.0 * dx_ij)
                penalty_grad_x[j] += c * (2.0 * dx_ij)
                penalty_grad_y[i] += c * (-2.0 * dy_ij)
                penalty_grad_y[j] += c * (2.0 * dy_ij)
                penalty_grad_r[i] += c * 2.0 * md
                penalty_grad_r[j] += c * 2.0 * md
    
    grad[:n] += 500.0 * penalty_grad_x
    grad[n:2*n] += 500.0 * penalty_grad_y
    grad[2*n:] += 500.0 * penalty_grad_r
    
    return _objective(x), grad


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.098
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    pos[25, 0] = 0.90
    pos[25, 1] = 0.90
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern with optimized spacing."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    y_spacing = r * np.sqrt(3)
    
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    for i in range(5):
        idx = i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r
        radii[idx] = r
    
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 2 * y_spacing
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + 3 * y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 4 * y_spacing
        radii[idx] = r
    
    best_score = -1
    best_pos = (0.5, 0.5)
    for cx in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
        for cy in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
            min_clearance = 1.0
            for i in range(25):
                dist = np.sqrt((cx - pos[i, 0])**2 + (cy - pos[i, 1])**2)
                clearance = dist - radii[i]
                min_clearance = min(min_clearance, clearance)
            min_clearance = min(min_clearance, cx, 1-cx, cy, 1-cy)
            if min_clearance > best_score:
                best_score = min_clearance
                best_pos = (cx, cy)
    
    pos[25, 0] = best_pos[0]
    pos[25, 1] = best_pos[1]
    radii[25] = max(best_score * 0.9, 0.02)
    
    return pos, radii


def _initial_center_ring():
    """Center circle surrounded by a ring of circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.12
    
    ring_r = 0.08
    center_r = 0.12
    ring_dist = center_r + ring_r
    
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    corner_r = 0.06
    corners = [
        (0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
        (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.3, 0.7)
    ]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(300):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision local search with multiple restarts to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Use analytical gradient for L-BFGS-B
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10), (1e-16, 1e-12)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 10000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Powell pass
    if time.time() < deadline - 2:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 4000, 'ftol': 1e-15})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Nelder-Mead fine tuning
    if time.time() < deadline - 1.5:
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration with time-aware allocation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B with analytical gradient
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 8000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Powell
    if time.time() < deadline - 6:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-13})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def _differential_evolution_run(deadline):
    """Run differential evolution for global search with time monitoring."""
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
    
    de_deadline = deadline - 25
    
    try:
        result = differential_evolution(
            _objective, bounds,
            maxiter=200, popsize=25,
            tol=1e-7, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=False,
            updating='deferred',
            callback=lambda xk, convergence: (time.time() > de_deadline)
        )
        
        if result.fun < 0:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    config_order = [0, 1, 5, 6, 2, 3, 4]
    for idx in config_order:
        if time.time() > deadline - 30:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    if time.time() < deadline - 35:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.095
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.075
        best_pos = pos
        best_radii = radii
    
    if time.time() < deadline - 3:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
