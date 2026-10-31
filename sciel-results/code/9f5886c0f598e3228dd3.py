"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, LinearConstraint

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


def _objective(x, penalty_weight=1000.0):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-4)
    
    penalty = 0.0
    # Boundary penalties (smooth)
    for i in range(n):
        penalty += max(0.0, radii[i] - pos[i, 0])**2
        penalty += max(0.0, pos[i, 0] - (1.0 - radii[i]))**2
        penalty += max(0.0, radii[i] - pos[i, 1])**2
        penalty += max(0.0, pos[i, 1] - (1.0 - radii[i]))**2
    
    # Pairwise overlap penalties (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            min_dist_sq = min_dist * min_dist
            if dist_sq < min_dist_sq:
                penalty += (min_dist_sq - dist_sq)**2 / (4.0 * min_dist_sq)
    
    return -np.sum(radii) + penalty_weight * penalty


def _objective_slsqp(x, penalty_weight=1000.0):
    """Objective for SLSQP with hard constraints handled separately."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    return -np.sum(radii)


def _slsqp_constraint(i, x, penalty_weight=1000.0):
    """Create a constraint function for circle i."""
    n = N
    def constraint(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        r = radii[i]
        # Boundary constraints
        c1 = pos[i, 0] - r
        c2 = 1.0 - pos[i, 0] - r
        c3 = pos[i, 1] - r
        c4 = 1.0 - pos[i, 1] - r
        return np.array([c1, c2, c3, c4])
    return constraint


def _slsqp_pairwise_constraint(i, j, x, penalty_weight=1000.0):
    """Create a pairwise non-overlap constraint function."""
    n = N
    def constraint(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - radii[i] - radii[j]
    return constraint


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
    x_offset = 0.09
    y_spacing = r * np.sqrt(3)
    
    for i in range(5):
        idx = i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset
        radii[idx] = r
    
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_offset + r + i * 2 * r
        pos[idx, 1] = x_offset + y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset + 2 * y_spacing
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_offset + r + i * 2 * r
        pos[idx, 1] = x_offset + 3 * y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset + 4 * y_spacing
        radii[idx] = r
    
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
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
        (0.5, 0.5)
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


def _initial_corner_heavy():
    """Create a corner-heavy initial configuration."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 large corner circles
    corner_r = 0.15
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (x, y) in enumerate(corners):
        pos[i, 0] = x
        pos[i, 1] = y
        radii[i] = corner_r
    
    # 12 edge circles
    edge_r = 0.08
    edges = [
        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),
        (0.3, 0.1), (0.7, 0.1), (0.3, 0.9), (0.7, 0.9),
        (0.1, 0.3), (0.1, 0.7), (0.9, 0.3), (0.9, 0.7)
    ]
    for i, (x, y) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = x
        pos[idx, 1] = y
        radii[idx] = edge_r
    
    # 10 center circles
    center_r = 0.07
    center_positions = [
        (0.5, 0.5), (0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6),
        (0.35, 0.5), (0.65, 0.5), (0.5, 0.35), (0.5, 0.65), (0.5, 0.5)
    ]
    for i, (x, y) in enumerate(center_positions):
        idx = 16 + i
        pos[idx, 0] = x
        pos[idx, 1] = y
        radii[idx] = center_r
    
    return pos, radii


def _polish_trust_constr(pos, radii, deadline):
    """Polish using trust-constr with hard constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 0.999999))
        bounds.append((1e-6, 0.999999))
    for i in range(N):
        bounds.append((1e-6, 0.3))
    
    A_boundary = np.zeros((4 * N, 4 * N))
    for i in range(N):
        A_boundary[4*i, 2*i] = 1.0
        A_boundary[4*i, 2*i+1] = -1.0
        A_boundary[4*i+1, 2*i] = -1.0
        A_boundary[4*i+1, 2*i+1] = -1.0
        A_boundary[4*i+2, 2*i+2] = 1.0
        A_boundary[4*i+2, 2*i+3] = -1.0
        A_boundary[4*i+3, 2*i+2] = -1.0
        A_boundary[4*i+3, 2*i+3] = -1.0
    
    boundary_constraint = LinearConstraint(A_boundary, lb=np.zeros(4 * N), ub=np.inf)
    
    n_pairs = N * (N - 1) // 2
    pairwise_nc = NonlinearConstraint(
        _pairwise_constraint_func,
        lb=np.zeros(n_pairs),
        ub=np.inf
    )
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    try:
        result = minimize(
            _trust_constr_objective,
            x0,
            method='trust-constr',
            jac=_trust_constr_jacobian,
            bounds=bounds,
            constraints=[boundary_constraint, pairwise_nc],
            options={'maxiter': 200, 'gtol': 1e-10, 'xtol': 1e-10}
        )
        
        if result.success or result.fun < -best_score + 1e-12:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-6)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _pairwise_constraint_func(x):
    """Pairwise non-overlap constraints: dist(i,j) >= r[i] + r[j]."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            constraints.append(dist - radii[i] - radii[j])
    return np.array(constraints)


def _trust_constr_objective(x):
    """Objective for trust-constr: minimize negative sum of radii."""
    n = N
    radii = x[2*n:]
    return -np.sum(radii)


def _trust_constr_jacobian(x):
    """Jacobian of the objective."""
    n = N
    jac = np.zeros_like(x)
    jac[2*n:] = -1.0
    return jac


def _polish_slsqp(pos, radii, deadline):
    """Polish using SLSQP with hard constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 0.999999))
        bounds.append((1e-6, 0.999999))
    for i in range(N):
        bounds.append((1e-6, 0.3))
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: _slsqp_constraint(i, x)[0]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: _slsqp_constraint(i, x)[1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: _slsqp_constraint(i, x)[2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: _slsqp_constraint(i, x)[3]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _slsqp_pairwise_constraint(i, j, x)})
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    try:
        result = minimize(
            _objective_slsqp,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-6)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _polish(pos, radii, deadline):
    """High-precision local search with multiple methods."""
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
    
    if time.time() < deadline - 10:
        best_pos, best_radii, best_score = _polish_trust_constr(best_pos, best_radii, deadline)
    
    if time.time() < deadline - 8:
        best_pos, best_radii, best_score = _polish_slsqp(best_pos, best_radii, deadline)
    
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10), (1e-16, 1e-12)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': ftol, 'gtol': gtol})
            
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
    
    if time.time() < deadline - 2:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-15})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_pp)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 1.5:
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
            
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
    
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            
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
    
    if time.time() < deadline - 6:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-13})
            
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
            maxiter=100, popsize=15,
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
    
    pos, radii = _initial_corner_heavy()
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
    
    config_order = [0, 1, 5, 6, 2, 3, 4, 7]
    for idx in config_order:
        if time.time() > deadline - 25:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    if time.time() < deadline - 30:
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
    
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
