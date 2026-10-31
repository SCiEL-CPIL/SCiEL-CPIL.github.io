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


def _slsqp_constraints(x):
    """Return list of constraint dicts for SLSQP: boundary and non-overlap."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    constraints = []
    
    # Boundary constraints: each circle must be inside [0,1]^2
    for i in range(n):
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i],
                           'jac': lambda x, i=i: _boundary_jac(x, i, 0)})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i],
                           'jac': lambda x, i=i: _boundary_jac(x, i, 1)})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i],
                           'jac': lambda x, i=i: _boundary_jac(x, i, 2)})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i],
                           'jac': lambda x, i=i: _boundary_jac(x, i, 3)})
    
    # Non-overlap: dist_ij >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _overlap_val(x, i, j),
                               'jac': lambda x, i=i, j=j: _overlap_jac(x, i, j)})
    
    return constraints


def _boundary_jac(x, i, which):
    """Jacobian for boundary constraint. which: 0=x-r, 1=1-x-r, 2=y-r, 3=1-y-r."""
    n = N
    jac = np.zeros(3*n)
    if which == 0:
        jac[2*i] = 1.0
        jac[2*n+i] = -1.0
    elif which == 1:
        jac[2*i] = -1.0
        jac[2*n+i] = -1.0
    elif which == 2:
        jac[2*i+1] = 1.0
        jac[2*n+i] = -1.0
    elif which == 3:
        jac[2*i+1] = -1.0
        jac[2*n+i] = -1.0
    return jac


def _overlap_val(x, i, j):
    """Non-overlap constraint value: dist_ij - r_i - r_j >= 0."""
    n = N
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - x[2*n+i] - x[2*n+j]


def _overlap_jac(x, i, j):
    """Jacobian for non-overlap constraint."""
    n = N
    jac = np.zeros(3*n)
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(dist_sq) if dist_sq > 1e-14 else 1e-7
    inv_dist = 1.0 / dist
    jac[2*i] = dx * inv_dist
    jac[2*i+1] = dy * inv_dist
    jac[2*j] = -dx * inv_dist
    jac[2*j+1] = -dy * inv_dist
    jac[2*n+i] = -1.0
    jac[2*n+j] = -1.0
    return jac


def _objective(x, beta=50.0):
    """Smooth objective: minimize negative sum of radii + soft penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Soft penalty for going outside square using log-sum-exp style
    penalty = 0.0
    for i in range(n):
        # Left boundary violation
        v = radii[i] - pos[i, 0]
        if v > 0:
            penalty += v * v
        # Right boundary violation
        v = pos[i, 0] - (1.0 - radii[i])
        if v > 0:
            penalty += v * v
        # Bottom boundary violation
        v = radii[i] - pos[i, 1]
        if v > 0:
            penalty += v * v
        # Top boundary violation
        v = pos[i, 1] - (1.0 - radii[i])
        if v > 0:
            penalty += v * v
    
    # Soft penalty for overlapping using smooth approximation
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            # Smooth penalty: use max(0, min_dist - dist)^2 but smoothed
            violation = min_dist - dist
            if violation > 0:
                # Quadratic penalty with smooth transition
                penalty += violation * violation
            elif violation > -0.01:
                # Small smooth region for gradient
                penalty += 0.0 * violation  # zero but keeps differentiability
    
    return -np.sum(radii) + beta * penalty


def _objective_smooth(x, beta=50.0, smooth_eps=0.01):
    """Fully smooth objective using smooth max for overlaps and boundaries."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Smooth boundary penalty using softplus-like function
    penalty = 0.0
    for i in range(n):
        for coord in range(2):
            # Distance to left/bottom boundary: d = pos - r, want d >= 0
            d = pos[i, coord] - radii[i]
            # Smooth penalty: -d + smooth_log(1 + exp(-d/smooth_eps)) approximates max(0,-d)
            # Use quadratic for large violations, smooth for small
            if d < 0:
                penalty += d * d
            # Distance to right/top boundary: d = 1 - pos - r, want d >= 0
            d = (1.0 - pos[i, coord]) - radii[i]
            if d < 0:
                penalty += d * d
    
    # Overlap penalty using smooth approximation
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq > 1e-14:
                dist = np.sqrt(dist_sq)
            else:
                dist = 0.0
            violation = min_dist - dist
            if violation > 0:
                penalty += violation * violation
            elif violation > -smooth_eps:
                # Smooth transition zone: use quadratic interpolation
                t = (violation + smooth_eps) / smooth_eps  # in [0, 1]
                penalty += t * t * smooth_eps * smooth_eps
    
    return -np.sum(radii) + beta * penalty


def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        # 5x5 grid, use 25 cells, add one more
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        # Add 26th circle in a gap
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
        # Hexagonal packing pattern
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        r = 0.1
        rows = 5
        for row in range(rows):
            y = 0.1 + row * r * np.sqrt(3)
            if row % 2 == 0:
                xs = np.arange(5) * 2 * r + r
            else:
                xs = np.arange(5) * 2 * r + 2 * r
            for col, x in enumerate(xs):
                idx = row * 5 + col
                if idx < N:
                    pos[idx] = [x, y]
                    radii[idx] = r
        return pos, radii
    
    elif strategy == 'center_ring':
        # One center, rings around
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        pos[0] = [0.5, 0.5]
        radii[0] = 0.15
        
        # Inner ring: 8 circles
        r_inner = 0.12
        dist_inner = 0.3
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
            radii[i+1] = r_inner
        
        # Outer ring: 17 circles
        r_outer = 0.08
        dist_outer = 0.42
        for i in range(17):
            angle = 2 * np.pi * i / 17
            pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
            radii[i+9] = r_outer
        
        return pos, radii
    
    elif strategy == 'random':
        # Random initial positions
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        # Default: slightly perturbed grid
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        # Check boundaries
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.98)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.98))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.98))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.98))
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = (dist / (min_dist + 1e-12)) * 0.98
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _slsqp_polish(pos, radii, deadline=None):
    """SLSQP polish with explicit constraints for tighter packing."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    constraints = _slsqp_constraints(x0)
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        if result.success or result.fun < -np.sum(radii):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _trust_constr_polish(pos, radii, deadline=None):
    """Trust-constr polish with explicit constraints."""
    from scipy.optimize import LinearConstraint, NonlinearConstraint
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    # Build constraint functions
    def boundary_fun(x):
        n = N
        vals = np.zeros(4*n)
        for i in range(n):
            vals[4*i] = x[2*i] - x[2*n+i]
            vals[4*i+1] = 1.0 - x[2*i] - x[2*n+i]
            vals[4*i+2] = x[2*i+1] - x[2*n+i]
            vals[4*i+3] = 1.0 - x[2*i+1] - x[2*n+i]
        return vals
    
    def boundary_jac(x):
        n = N
        jac = np.zeros((4*n, 3*n))
        for i in range(n):
            jac[4*i, 2*i] = 1.0
            jac[4*i, 2*n+i] = -1.0
            jac[4*i+1, 2*i] = -1.0
            jac[4*i+1, 2*n+i] = -1.0
            jac[4*i+2, 2*i+1] = 1.0
            jac[4*i+2, 2*n+i] = -1.0
            jac[4*i+3, 2*i+1] = -1.0
            jac[4*i+3, 2*n+i] = -1.0
        return jac
    
    # Non-overlap constraints
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    n_pairs = len(pairs)
    
    def overlap_fun(x):
        n = N
        vals = np.zeros(n_pairs)
        for k, (i, j) in enumerate(pairs):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            vals[k] = dist - x[2*n+i] - x[2*n+j]
        return vals
    
    def overlap_jac(x):
        n = N
        jac = np.zeros((n_pairs, 3*n))
        for k, (i, j) in enumerate(pairs):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-14 else 1e-7
            inv = 1.0 / dist
            jac[k, 2*i] = dx * inv
            jac[k, 2*i+1] = dy * inv
            jac[k, 2*j] = -dx * inv
            jac[k, 2*j+1] = -dy * inv
            jac[k, 2*n+i] = -1.0
            jac[k, 2*n+j] = -1.0
        return jac
    
    try:
        nl_constraints = [
            NonlinearConstraint(boundary_fun, 0, np.inf, jac=boundary_jac),
            NonlinearConstraint(overlap_fun, 0, np.inf, jac=overlap_jac)
        ]
        
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='trust-constr',
                          bounds=bounds, constraints=nl_constraints,
                          options={'maxiter': 100, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0})
        if result.fun < -np.sum(radii):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _growth_pass(pos, radii, deadline=None):
    """Try to grow each circle individually to fill available space."""
    pos = pos.copy()
    radii = radii.copy()
    improved = True
    n_iter = 0
    
    while improved and n_iter < 80:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        n_iter += 1
        for i in range(N):
            # Compute max possible radius for circle i given others fixed
            max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            max_r = max(0.001, max_r * 0.9995)
            if max_r > radii[i] + 1e-7:
                radii[i] = max_r
                improved = True
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp


def _polish(pos, radii, deadline=None):
    """High-precision polish using multiple methods."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    best_pos, best_radii = pos.copy(), radii.copy()
    best_score = np.sum(radii)
    
    # Try L-BFGS-B with smooth objective
    try:
        result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 100})
        if result.fun < _objective_smooth(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
                best_score = np.sum(radii_pp)
                best_pos, best_radii = pos_pp, radii_pp
    except Exception:
        pass
    
    # Try SLSQP with explicit constraints
    if deadline is None or time.time() < deadline - 8:
        try:
            pos_s, rad_s = _slsqp_polish(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_s, rad_s) and np.sum(rad_s) > best_score:
                best_score = np.sum(rad_s)
                best_pos, best_radii = pos_s, rad_s
        except Exception:
            pass
    
    # Try trust-constr
    if deadline is None or time.time() < deadline - 8:
        try:
            pos_t, rad_t = _trust_constr_polish(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_t, rad_t) and np.sum(rad_t) > best_score:
                best_score = np.sum(rad_t)
                best_pos, best_radii = pos_t, rad_t
        except Exception:
            pass
    
    # Final growth pass
    if deadline is None or time.time() < deadline - 3:
        try:
            pos_g, rad_g = _growth_pass(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_g, rad_g) and np.sum(rad_g) > best_score:
                best_score = np.sum(rad_g)
                best_pos, best_radii = pos_g, rad_g
        except Exception:
            pass
    
    return best_pos, best_radii


def _force_directed_anneal(pos, radii, n_iter=500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Initialize with small random velocities
    vel = np.zeros_like(pos)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule: slower cooling
        temp = 0.015 * np.exp(-0.0015 * it)
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces (smooth push inward)
        for i in range(N):
            # Left
            margin = radii[i] + 0.005
            if pos[i, 0] < margin:
                forces[i, 0] += 8.0 * (margin - pos[i, 0])
            # Right
            if pos[i, 0] > 1.0 - margin:
                forces[i, 0] -= 8.0 * (pos[i, 0] - (1.0 - margin))
            # Bottom
            if pos[i, 1] < margin:
                forces[i, 1] += 8.0 * (margin - pos[i, 1])
            # Top
            if pos[i, 1] > 1.0 - margin:
                forces[i, 1] -= 8.0 * (pos[i, 1] - (1.0 - margin))
        
        # Repulsion forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    # Strong repulsion
                    overlap = min_dist - dist
                    force_mag = 12.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.3 and dist > 1e-6:
                    # Weak repulsion to encourage separation
                    force_mag = 0.8 * (min_dist * 1.3 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions with velocity damping and random perturbation
        vel = 0.85 * vel + 0.15 * forces
        pos += vel * 0.008 + np.random.normal(0, temp, size=pos.shape)
        
        # Clamp to valid range
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.005, 0.995)
            pos[i, 1] = np.clip(pos[i, 1], 0.005, 0.995)
        
        # Try to grow radii slightly
        for i in range(N):
            max_grow = 0.001
            # Boundary constraints
            max_grow = min(max_grow, pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            # Overlap constraints
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0001:
                radii[i] += max_grow * 0.7
        
        # Check validity and update best
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    
    # Phase 1: Multi-start local optimization
    for strategy in strategies:
        if time.time() > deadline - 55:
            break
        
        for seed in range(5):
            if time.time() > deadline - 50:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            
            # Combine into single vector
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0.005, 0.995))
                bounds.append((0.005, 0.995))
            for i in range(N):
                bounds.append((0.005, 0.35))
            
            # Try L-BFGS-B with smooth objective
            try:
                result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-9})
                if result.fun < _objective_smooth(x0):
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    # Post-process to ensure validity
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
            
            # Try Nelder-Mead
            if time.time() > deadline - 45:
                break
            
            try:
                x_start = np.concatenate([pos.flatten(), radii])
                result_nm = minimize(_objective_smooth, x_start, method='Nelder-Mead',
                                     options={'maxiter': 1500, 'fatol': 1e-11})
                if result_nm.fun < _objective_smooth(x_start):
                    x_opt = result_nm.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Phase 2: Force-directed annealing from best result
    if best_pos is not None and time.time() < deadline - 35:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=1000, deadline=deadline - 30)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish with constrained methods
    if best_pos is not None and time.time() < deadline - 12:
        try:
            pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 5)
            score_pol = np.sum(radii_pol)
            if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
        except Exception:
            pass
    
    # Phase 4: Additional SLSQP + growth from best
    if best_pos is not None and time.time() < deadline - 6:
        try:
            pos_s2, rad_s2 = _slsqp_polish(best_pos, best_radii, deadline=deadline - 4)
            if _is_valid(pos_s2, rad_s2):
                pos_g2, rad_g2 = _growth_pass(pos_s2, rad_s2, deadline=deadline - 2)
                score_g2 = np.sum(rad_g2)
                if _is_valid(pos_g2, rad_g2) and score_g2 > best_score:
                    best_score = score_g2
                    best_pos = pos_g2.copy()
                    best_radii = rad_g2.copy()
        except Exception:
            pass
    
    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
