"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _slsqp_constraints(pos, radii):
    """Compute SLSQP constraint values (>=0 means satisfied)."""
    n = len(radii)
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    bc = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    # Overlap constraints: dist - (ri + rj) >= 0
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    oc = dist[mask] - min_dist[mask]
    return np.concatenate([bc, oc])

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Smooth boundary violations
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)
    
    boundary_pen = np.sum(_soft_max(left_v, beta)) + np.sum(_soft_max(right_v, beta)) + \
                   np.sum(_soft_max(bottom_v, beta)) + np.sum(_soft_max(top_v, beta))
    
    # Smooth overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """
    Simulated annealing: perturb positions and radii, accept based on energy.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Current energy (penalty-based)
    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 50.0)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Perturb: move one circle slightly
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)
        
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr
        
        # Clamp
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)
        
        # Try the move
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 50.0)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            # Reject
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        # Periodically check validity and update best
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_constraint_func(x, i):
    """SLSQP constraint for circle i: boundary + overlaps with all j != i."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    # Boundary
    bc = np.array([
        pos[i, 0] - radii[i],
        1.0 - pos[i, 0] - radii[i],
        pos[i, 1] - radii[i],
        1.0 - pos[i, 1] - radii[i]
    ])
    # Overlaps
    oc = np.array([])
    for j in range(N):
        if j != i:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            oc = np.append(oc, d - radii[i] - radii[j])
    return np.concatenate([bc, oc])

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP (constraints) and L-BFGS-B."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
    
    best_x = x.copy()
    best_score = np.sum(radii)
    
    # SLSQP with explicit constraints
    if time.time() < deadline - 10:
        try:
            constraints = []
            for i in range(N):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: _slsqp_constraint_func(x, i)})
            
            result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                              args=(100.0,), constraints=constraints,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.success or result.fun < _objective(x, 100.0) - 1e-10:
                pos_t = result.x[:2*N].reshape(N, 2)
                rad_t = np.maximum(result.x[2*N:], 1e-4)
                pos_pp, rad_pp = _post_process(pos_t, rad_t)
                if _is_valid(pos_pp, rad_pp) and np.sum(rad_pp) > best_score:
                    best_score = np.sum(rad_pp)
                    best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass
    
    # L-BFGS-B with high penalty
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, best_x, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            pos_t = result.x[:2*N].reshape(N, 2)
            rad_t = np.maximum(result.x[2*N:], 1e-4)
            pos_pp, rad_pp = _post_process(pos_t, rad_t)
            if _is_valid(pos_pp, rad_pp) and np.sum(rad_pp) > best_score:
                best_score = np.sum(rad_pp)
                best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass
    
    # Perturbation-restart polish: try small random perturbations and re-optimize
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            pert = best_x.copy()
            # Small perturbation
            pert[:2*N] += rng.normal(0, 0.003, 2*N)
            pert[2*N:] += rng.normal(0, 0.001, N)
            pert = np.clip(pert, [0.005]*2*N + [0.005]*N, [0.995]*2*N + [0.5]*N)
            
            try:
                result = minimize(_objective, pert, method='L-BFGS-B', bounds=bounds,
                                  args=(10000.0,),
                                  options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11})
                pos_t = result.x[:2*N].reshape(N, 2)
                rad_t = np.maximum(result.x[2*N:], 1e-4)
                pos_pp, rad_pp = _post_process(pos_t, rad_t)
                if _is_valid(pos_pp, rad_pp) and np.sum(rad_pp) > best_score:
                    best_score = np.sum(rad_pp)
                    best_x = np.concatenate([pos_pp.flatten(), rad_pp])
            except Exception:
                pass
    
    pos_out = best_x[:2*N].reshape(N, 2)
    radii_out = np.maximum(best_x[2*N:], 1e-4)
    
    return pos_out, radii_out

def _grad_descent(pos, radii, lr=0.01, n_steps=500, deadline=None):
    """
    Gradient descent with analytic gradients for faster convergence.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute gradients
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap violations (squared)
        viol_sq = np.maximum(0.0, min_dist**2 - dist_sq)
        # Only use upper triangle to avoid double counting
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        viol_sq = viol_sq * mask
        
        # Gradient for position
        grad_pos = np.zeros((n, 2))
        for i in range(n):
            for j in range(i+1, n):
                if viol_sq[i, j] > 0:
                    d = dist[i, j]
                    if d > 1e-8:
                        # d/dx_i of (ri+rj)^2 - (xi-xj)^2-(yi-yj)^2 = -2(xi-xj)
                        # We want to minimize violation, so gradient points to reduce viol
                        # viol = (ri+rj)^2 - dist^2, d(viol)/d(xi) = -2(xi-xj)
                        # We add penalty * viol, so gradient of penalty wrt xi = -2(xi-xj) * 2 * viol / ... 
                        # Actually for quadratic penalty: pen = sum viol^2, d(pen)/d(xi) = 2*viol * d(viol)/d(xi) = 2*viol * (-2*(xi-xj))
                        # But we want to move in direction that reduces pen, so update = -lr * grad
                        # grad_pos[i] += 2*viol * (-2*(pos[i]-pos[j])/dist) ... let me just use simple repulsion
                        grad_pos[i, 0] += -2.0 * viol_sq[i, j] * (pos[i, 0] - pos[j, 0]) / max(d, 1e-8)
                        grad_pos[i, 1] += -2.0 * viol_sq[i, j] * (pos[i, 1] - pos[j, 1]) / max(d, 1e-8)
                        grad_pos[j, 0] -= -2.0 * viol_sq[i, j] * (pos[i, 0] - pos[j, 0]) / max(d, 1e-8)
                        grad_pos[j, 1] -= -2.0 * viol_sq[i, j] * (pos[i, 1] - pos[j, 1]) / max(d, 1e-8)
        
        # Gradient for radius: we want to maximize sum(r), so base gradient is -1
        grad_r = np.ones(n)
        for i in range(n):
            for j in range(i+1, n):
                if viol_sq[i, j] > 0:
                    grad_r[i] += 2.0 * viol_sq[i, j]
                    grad_r[j] += 2.0 * viol_sq[i, j]
        
        # Boundary gradients
        for i in range(n):
            if pos[i, 0] - radii[i] < 0:
                grad_pos[i, 0] -= 2.0 * (radii[i] - pos[i, 0])
                grad_r[i] += 2.0 * (radii[i] - pos[i, 0])
            if pos[i, 0] + radii[i] > 1:
                grad_pos[i, 0] += 2.0 * (pos[i, 0] + radii[i] - 1)
                grad_r[i] += 2.0 * (pos[i, 0] + radii[i] - 1)
            if pos[i, 1] - radii[i] < 0:
                grad_pos[i, 1] -= 2.0 * (radii[i] - pos[i, 1])
                grad_r[i] += 2.0 * (radii[i] - pos[i, 1])
            if pos[i, 1] + radii[i] > 1:
                grad_pos[i, 1] += 2.0 * (pos[i, 1] + radii[i] - 1)
                grad_r[i] += 2.0 * (pos[i, 1] + radii[i] - 1)
        
        # Adaptive learning rate
        effective_lr = lr * (1.0 + 0.5 * np.cos(np.pi * step / max(n_steps, 1)))
        pos -= effective_lr * grad_pos
        radii -= effective_lr * grad_r
        
        # Clamp
        pos = np.clip(pos, 0.005, 0.995)
        radii = np.clip(radii, 0.005, 0.5)
        
        # Check validity periodically
        if step % 30 == 0 or step == n_steps - 1:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _local_optimize(pos, radii, deadline):
    """
    Local optimization: try to increase each radius by pushing neighbors.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try to increase each radius
    for i in range(n):
        if time.time() > deadline:
            break
            
        # Try increasing radius of circle i
        old_r = radii[i]
        for delta_r in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            radii[i] = old_r + delta_r
            
            # Check if still valid, if not, try to fix
            valid = True
            for j in range(n):
                if j == i:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j]:
                    valid = False
                    break
            
            if valid and _is_valid(pos, radii):
                best_score = np.sum(radii)
                best_pos = pos.copy()
                best_radii = radii.copy()
            else:
                radii[i] = old_r
                break
        
        # Try moving circle i to a better position
        for _ in range(30):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(np.random.randint(100000))
            dx = rng.normal(0, 0.01)
            dy = rng.normal(0, 0.01)
            new_x = pos[i, 0] + dx
            new_y = pos[i, 1] + dy
            
            if 0.01 <= new_x <= 0.99 and 0.01 <= new_y <= 0.99:
                old_x, old_y = pos[i, 0], pos[i, 1]
                pos[i, 0] = new_x
                pos[i, 1] = new_y
                
                if _is_valid(pos, radii):
                    # Try to increase radius
                    for delta_r in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                        radii[i] += delta_r
                        if _is_valid(pos, radii):
                            score = np.sum(radii)
                            if score > best_score:
                                best_score = score
                                best_pos = pos.copy()
                                best_radii = radii.copy()
                        else:
                            radii[i] -= delta_r
                            break
                else:
                    pos[i, 0] = old_x
                    pos[i, 1] = old_y
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
        
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective with L-BFGS-B
        for p_w in penalty_stages:
            if time.time() > deadline - 12:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Smooth objective
        if time.time() < deadline - 12:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 8:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Gradient descent
        if time.time() < deadline - 18:
            try:
                pos_gd, rad_gd = _grad_descent(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    lr=0.008, n_steps=400, deadline=deadline - 14
                )
                pos_pp, radii_pp = _post_process(pos_gd, rad_gd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 4: Simulated annealing
        if time.time() < deadline - 10:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=1500, t0=0.03, t_end=0.0005,
                    seed=idx * 100, deadline=deadline - 7
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 5: Local optimization
        if time.time() < deadline - 6:
            try:
                pos_lo, rad_lo = _local_optimize(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    deadline=deadline - 4
                )
                pos_pp, radii_pp = _post_process(pos_lo, rad_lo)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish with SLSQP constraints + perturbation restarts
    if time.time() < deadline - 8:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final local optimization to squeeze radii
    if time.time() < deadline - 3:
        try:
            pos_lo, rad_lo = _local_optimize(best_pos, best_radii, deadline - 1)
            pos_pp, radii_pp = _post_process(pos_lo, rad_lo)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final tight polish
    if time.time() < deadline - 1:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Try harder to fix validity
        for _ in range(10):
            best_pos, best_radii = _post_process(best_pos, best_radii)
            if _is_valid(best_pos, best_radii):
                break
        
        if not _is_valid(best_pos, best_radii):
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N - 1:
                        pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                        radii[k] = 0.095
                        k += 1
            pos[k] = [0.5, 0.02]
            radii[k] = 0.01
            best_pos, best_radii = pos, radii
            best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
