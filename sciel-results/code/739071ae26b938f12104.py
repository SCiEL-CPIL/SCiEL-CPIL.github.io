"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint

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


def _pairwise_indices(n):
    """Precompute pairwise indices."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return np.array(pairs)

PAIRS = _pairwise_indices(N)


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-5)
    
    # Boundary violations
    boundary_viol = np.zeros(4*n)
    for i in range(n):
        boundary_viol[4*i]   = max(0.0, radii[i] - pos[i, 0])
        boundary_viol[4*i+1] = max(0.0, pos[i, 0] - (1.0 - radii[i]))
        boundary_viol[4*i+2] = max(0.0, radii[i] - pos[i, 1])
        boundary_viol[4*i+3] = max(0.0, pos[i, 1] - (1.0 - radii[i]))
    
    # Pairwise violations
    m = len(PAIRS)
    pair_viol = np.zeros(m)
    for idx, (i, j) in enumerate(PAIRS):
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i] + radii[j]
        pair_viol[idx] = max(0.0, min_dist - dist)
    
    # Log-sum-exp penalty
    tau = 0.005
    def _lse_penalty(v):
        if np.max(v) < 1e-12:
            return 0.0
        v_scaled = v / tau
        v_max = np.max(v_scaled)
        lse = v_max + np.log(np.sum(np.exp(v_scaled - v_max)))
        return lse * tau
    
    boundary_pen = _lse_penalty(boundary_viol)
    pair_pen = _lse_penalty(pair_viol)
    
    return -np.sum(radii) + 100.0 * boundary_pen + 100.0 * pair_pen


def _objective_hard(x):
    """Hard-constrained objective for SLSQP: just negative sum of radii."""
    n = N
    radii = x[2*n:]
    return -np.sum(radii)


def _boundary_constraint(x):
    """Boundary constraints: each circle must be inside [0,1]^2.
    Returns 4n values, each >= 0 for validity."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    c = np.zeros(4*n)
    for i in range(n):
        c[4*i]   = pos[i, 0] - radii[i]          # left boundary
        c[4*i+1] = (1.0 - pos[i, 0]) - radii[i]  # right boundary
        c[4*i+2] = pos[i, 1] - radii[i]          # bottom boundary
        c[4*i+3] = (1.0 - pos[i, 1]) - radii[i]  # top boundary
    return c


def _pairwise_constraint(x):
    """Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j for all pairs.
    Returns m values, each >= 0 for validity."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    m = len(PAIRS)
    c = np.zeros(m)
    for idx, (i, j) in enumerate(PAIRS):
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        c[idx] = dist - (radii[i] + radii[j])
    return c


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.095
    
    pos[25, 0] = 0.5
    pos[25, 1] = 0.05
    radii[25] = 0.045
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    dx = 2 * r
    dy = r * np.sqrt(3)
    
    for i in range(5):
        idx = i
        pos[idx, 0] = r + i * dx
        pos[idx, 1] = r
        radii[idx] = r
    
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = r + r + i * dx
        pos[idx, 1] = r + dy
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = r + i * dx
        pos[idx, 1] = r + 2 * dy
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = r + r + i * dx
        pos[idx, 1] = r + 3 * dy
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = r + i * dx
        pos[idx, 1] = r + 4 * dy
        radii[idx] = r
    
    # 26th circle in remaining space
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.04
    
    return pos, radii


def _initial_corner_heavy():
    """Corner-heavy initial configuration with larger corner circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners with larger radii
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i in range(4):
        pos[i, 0] = corners[i][0]
        pos[i, 1] = corners[i][1]
        radii[i] = 0.14
    
    # Edge midpoints
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i in range(4):
        idx = 4 + i
        pos[idx, 0] = edges[i][0]
        pos[idx, 1] = edges[i][1]
        radii[idx] = 0.10
    
    # Inner ring
    inner = [
        (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
        (0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25),
        (0.5, 0.75), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75),
        (0.75, 0.75), (0.4, 0.55), (0.6, 0.45), (0.55, 0.6),
        (0.45, 0.4), (0.6, 0.55), (0.4, 0.6), (0.55, 0.4),
        (0.45, 0.6), (0.6, 0.4), (0.5, 0.35), (0.35, 0.5),
        (0.65, 0.5), (0.5, 0.65)
    ]
    for i in range(N - 8):
        idx = 8 + i
        pos[idx, 0] = inner[i][0]
        pos[idx, 1] = inner[i][1]
        radii[idx] = 0.07
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively fix violations by shrinking radii minimally."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        # Boundary fixes
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r - 1e-10:
                radii[i] = max(max_r - 1e-10, 1e-5)
                valid = False
        
        # Pairwise fixes: shrink the larger circle more
        for idx, (i, j) in enumerate(PAIRS):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist - 1e-10:
                # Shrink both proportionally to their radii
                overlap = min_dist - dist
                if radii[i] + radii[j] < 1e-10:
                    continue
                scale_i = radii[i] / (radii[i] + radii[j])
                scale_j = radii[j] / (radii[i] + radii[j])
                radii[i] -= overlap * scale_i * 0.5
                radii[j] -= overlap * scale_j * 0.5
                radii[i] = max(radii[i], 1e-5)
                radii[j] = max(radii[j], 1e-5)
                valid = False
        
        if valid:
            break
    
    return pos, radii


def _polish_hard(pos, radii, deadline):
    """High-precision polish using SLSQP with hard constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-4, 1.0 - 1e-4))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    # Define hard constraints
    boundary_con = NonlinearConstraint(_boundary_constraint, 0.0, np.inf,
                                        jac='2-point', hess='2-point')
    pairwise_con = NonlinearConstraint(_pairwise_constraint, 0.0, np.inf,
                                        jac='2-point', hess='2-point')
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Pass 1: SLSQP
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective_hard, x0, method='SLSQP', bounds=bounds,
                              constraints=[boundary_con, pairwise_con],
                              options={'maxiter': 1000, 'ftol': 1e-15, 'iprint': 0})
            
            if result.success or result.fun < _objective_hard(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Pass 2: L-BFGS-B with penalty
    if time.time() < deadline - 5:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Pass 3: Another SLSQP pass
    if time.time() < deadline - 2:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective_hard, x0, method='SLSQP', bounds=bounds,
                              constraints=[boundary_con, pairwise_con],
                              options={'maxiter': 500, 'ftol': 1e-16, 'iprint': 0})
            
            if result.fun < _objective_hard(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.30))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Pass 1: L-BFGS-B with penalty (fast)
    if time.time() < deadline - 15:
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Pass 2: SLSQP with hard constraints (more precise)
    if time.time() < deadline - 15:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            
            boundary_con = NonlinearConstraint(_boundary_constraint, 0.0, np.inf,
                                                jac='2-point', hess='2-point')
            pairwise_con = NonlinearConstraint(_pairwise_constraint, 0.0, np.inf,
                                                jac='2-point', hess='2-point')
            
            result = minimize(_objective_hard, x0, method='SLSQP', bounds=bounds,
                              constraints=[boundary_con, pairwise_con],
                              options={'maxiter': 800, 'ftol': 1e-14, 'iprint': 0})
            
            if result.fun < _objective_hard(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Pass 3: Powell (derivative-free fine-tuning)
    if time.time() < deadline - 10:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final hard-constraint polish
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _polish_hard(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456, 789, 321]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.015, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.015, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Another perturbed corner-heavy
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.01, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each initial config
    n_configs = len(init_configs)
    time_per_config = min(18.0, (deadline - start_time - 25) / max(1, n_configs))
    
    for idx, (pos, radii) in enumerate(init_configs):
        if time.time() > deadline - 25:
            break
        
        config_deadline = min(deadline - 12, time.time() + time_per_config)
        pos_opt, radii_opt, score = _optimize_from(pos, radii, config_deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        pos, radii = _post_process(pos, radii)
        best_pos = pos
        best_radii = radii
        best_score = np.sum(radii)
    
    # Final post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 3:
        best_pos, best_radii, best_score = _polish_hard(best_pos, best_radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
