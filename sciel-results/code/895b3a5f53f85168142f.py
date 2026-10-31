"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
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
    for _ in range(300):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _try_radius_expansion(pos, radii, max_iterations=300):
    """Try to expand each circle's radius while maintaining validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])
            new_r = max_r_circ * 0.999999
            if new_r > r * 1.00001 and new_r > 0:
                old_r = radii[i]
                radii[i] = new_r
                if _is_valid(pos, radii):
                    improved = True
                else:
                    radii[i] = old_r
        if not improved:
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
        for _ in range(8000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r and max_r_circ > 0.002:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        if not found or best_r < 0.002:
            best_x, best_y = 0.5, 0.5
            best_r = 0.002
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.96
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
        placed = False
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.015:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.93
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.008
            k += 1
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.14, 0.14), (0.86, 0.14), (0.14, 0.86), (0.86, 0.86)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.17
    for i in range(4, N):
        if rng.random() < 0.5:
            edge = rng.randint(4)
            t = rng.uniform(0.15, 0.85)
            if edge == 0:
                x, y = t, 0.07
            elif edge == 1:
                x, y = t, 0.93
            elif edge == 2:
                x, y = 0.07, t
            else:
                x, y = 0.93, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        if max_r < 0.01:
            for attempt in range(300):
                x2 = rng.uniform(0.03, 0.97)
                y2 = rng.uniform(0.03, 0.97)
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
        radii[i] = max(max_r, 0.008) * 0.94
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.08, 0.92, (N, 2))
    radii = rng.uniform(0.04, 0.13, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_relax(pos, radii, n_iter=5000, dt=0.002, deadline=None):
    """
    Force-directed relaxation: circles repel each other and are attracted
    to expand their radii. This is a physics-based approach that naturally
    finds local optima without gradient computation.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(n_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < 1e-8:
                    dist = 1e-8
                    dx, dy = 0.01, 0.01
                elif dist < min_dist:
                    # Overlapping: strong repulsion
                    overlap = min_dist - dist
                    f = 50.0 * overlap / dist
                    forces[i, 0] += f * dx / dist
                    forces[i, 1] += f * dy / dist
                    forces[j, 0] -= f * dx / dist
                    forces[j, 1] -= f * dy / dist
                else:
                    # Slight attraction when close to touching (to pack tightly)
                    gap = dist - min_dist
                    if gap < 0.05:
                        f = -2.0 * gap / (gap + 0.01)
                        forces[i, 0] += f * dx / dist
                        forces[i, 1] += f * dy / dist
                        forces[j, 0] -= f * dx / dist
                        forces[j, 1] -= f * dy / dist
        
        # Wall forces: push circles away from walls
        for i in range(n):
            x, y = pos[i]
            r = radii[i]
            if x - r < 0:
                forces[i, 0] += 20.0 * (r - x)
            if x + r > 1:
                forces[i, 0] -= 20.0 * (x + r - 1)
            if y - r < 0:
                forces[i, 1] += 20.0 * (r - y)
            if y + r > 1:
                forces[i, 1] -= 20.0 * (y + r - 1)
        
        # Update positions
        for i in range(n):
            pos[i, 0] += dt * forces[i, 0]
            pos[i, 1] += dt * forces[i, 1]
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Try to grow radii
        for i in range(n):
            x, y = pos[i]
            r = radii[i]
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])
            if max_r_circ > r * 1.001 and max_r_circ > 0:
                radii[i] = min(r * 1.005, max_r_circ * 0.9999)
        
        # Check and update best
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_with_constraints(pos, radii, deadline, penalty_weight=1000.0):
    """
    SLSQP optimization with explicit hard constraints for non-overlap and boundaries.
    This is structurally different from penalty methods: constraints are enforced
    directly by the solver rather than through a penalty term.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.45))
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    # Hard constraints: boundary
    cons = []
    for i in range(N):
        # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N + i]})
        # x_i + r_i <= 1
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N + i]})
        # y_i + r_i <= 1
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    
    # Hard constraints: non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                        (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - 
                        (x[2*N + i] + x[2*N + j])**2})
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': 200, 'ftol': 1e-10})
        if result.success or result.fun < -np.sum(radii):
            pos_out = result.x[:2*N].reshape(N, 2)
            radii_out = np.maximum(result.x[2*N:], 1e-5)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def _slsqp_smooth(pos, radii, deadline, beta=30.0):
    """
    SLSQP with smooth epigraph formulation. Uses auxiliary variables to
    represent the non-smooth max(0, violation) terms smoothly.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.45))
    
    def objective(x):
        pos_x = x[:2*N].reshape(N, 2)
        rad_x = np.maximum(x[2*N:], 1e-6)
        
        # Boundary violations (smooth)
        bv = np.maximum(0.0, rad_x - pos_x[:, 0]) + np.maximum(0.0, pos_x[:, 0] - (1.0 - rad_x)) + \
             np.maximum(0.0, rad_x - pos_x[:, 1]) + np.maximum(0.0, pos_x[:, 1] - (1.0 - rad_x))
        
        # Overlap violations (smooth)
        dx = pos_x[:, 0, np.newaxis] - pos_x[np.newaxis, :, 0]
        dy = pos_x[:, 1, np.newaxis] - pos_x[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = rad_x[:, np.newaxis] + rad_x[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol = np.maximum(0.0, min_dist_sq - dist_sq)
        
        penalty = 100.0 * (np.sum(bv**2) + np.sum(viol[mask]**2))
        return -np.sum(rad_x) + penalty
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < objective(x0) - 1e-12:
            pos_out = result.x[:2*N].reshape(N, 2)
            radii_out = np.maximum(result.x[2*N:], 1e-5)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def _lbfgs_smooth(pos, radii, deadline, penalty_weight=500.0):
    """L-BFGS-B with smooth penalty objective."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.45))
    
    def objective(x):
        pos_x = x[:2*N].reshape(N, 2)
        rad_x = np.maximum(x[2*N:], 1e-6)
        
        bv = np.maximum(0.0, rad_x - pos_x[:, 0]) + np.maximum(0.0, pos_x[:, 0] - (1.0 - rad_x)) + \
             np.maximum(0.0, rad_x - pos_x[:, 1]) + np.maximum(0.0, pos_x[:, 1] - (1.0 - rad_x))
        
        dx = pos_x[:, 0, np.newaxis] - pos_x[np.newaxis, :, 0]
        dy = pos_x[:, 1, np.newaxis] - pos_x[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = rad_x[:, np.newaxis] + rad_x[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol = np.maximum(0.0, min_dist_sq - dist_sq)
        
        penalty = penalty_weight * (np.sum(bv**2) + np.sum(viol[mask]**2))
        return -np.sum(rad_x) + penalty
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < objective(x0) - 1e-12:
            pos_out = result.x[:2*N].reshape(N, 2)
            radii_out = np.maximum(result.x[2*N:], 1e-5)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=5000, t0=0.04, t_end=0.0005, seed=0, deadline=None):
    """Simulated annealing with position and radius perturbations."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    def energy(p, r):
        # Simple energy: negative sum of radii + penalty for violations
        bv = np.maximum(0.0, r - p[:, 0]) + np.maximum(0.0, p[:, 0] - (1.0 - r)) + \
             np.maximum(0.0, r - p[:, 1]) + np.maximum(0.0, p[:, 1] - (1.0 - r))
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        viol = np.maximum(0.0, min_dist_sq - dist_sq)
        return -np.sum(r) + 50.0 * (np.sum(bv**2) + np.sum(viol[mask]**2))
    
    current_energy = energy(pos, radii)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = np.clip(old_x + dx, 0.01, 0.99)
        pos[i, 1] = np.clip(old_y + dy, 0.01, 0.99)
        radii[i] = np.clip(old_r + dr, 0.005, 0.4)
        
        new_energy = energy(pos, radii)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _perturbation_local_search(pos, radii, n_perturbations=300, max_step=0.015, seed=0, deadline=None):
    """Local search with random perturbations."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for p in range(n_perturbations):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(n, size=n_pert, replace=False)
        
        old_pos = pos[indices].copy()
        old_radii = radii[indices].copy()
        
        for idx in indices:
            dx = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dy = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dr = rng.normal(0, max_step * 0.3)
            
            pos[idx, 0] = np.clip(pos[idx, 0] + dx, 0.005, 0.995)
            pos[idx, 1] = np.clip(pos[idx, 1] + dy, 0.005, 0.995)
            radii[idx] = np.clip(radii[idx] + dr, 0.005, 0.4)
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = best_pos.copy()
                radii = best_radii.copy()
        else:
            pos[indices] = old_pos
            radii[indices] = old_radii
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """Multi-stage polishing."""
    pos_out = pos.copy()
    radii_out = radii.copy()
    
    # Stage 1: L-BFGS-B smooth
    if time.time() < deadline - 20:
        p, r = _lbfgs_smooth(pos_out, radii_out, deadline, penalty_weight=500.0)
        if np.sum(r) > np.sum(radii_out):
            pos_out, radii_out = p, r
    
    # Stage 2: Force-directed relaxation
    if time.time() < deadline - 15:
        p, r = _force_directed_relax(pos_out, radii_out, n_iter=3000, dt=0.0015, deadline=deadline)
        if np.sum(r) > np.sum(radii_out):
            pos_out, radii_out = p, r
    
    # Stage 3: SLSQP smooth
    if time.time() < deadline - 10:
        p, r = _slsqp_smooth(pos_out, radii_out, deadline)
        if np.sum(r) > np.sum(radii_out):
            pos_out, radii_out = p, r
    
    # Stage 4: Radius expansion
    if time.time() < deadline - 5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=200)
    
    # Stage 5: Perturbation local search
    if time.time() < deadline - 3:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out, n_perturbations=150, max_step=0.008, seed=42, deadline=deadline)
    
    # Stage 6: Final radius expansion
    if time.time() < deadline - 1:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=100)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
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
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        # Force-directed relaxation first (fast, no gradient)
        if time.time() < deadline - 15:
            p, r = _force_directed_relax(pos_init, rad_init, n_iter=4000, dt=0.002, deadline=deadline)
            if _is_valid(p, r) and np.sum(r) > best_score:
                best_score = np.sum(r)
                best_pos = p.copy()
                best_radii = r.copy()
        
        # L-BFGS-B smooth
        if time.time() < deadline - 15:
            p, r = _lbfgs_smooth(pos_init, rad_init, deadline, penalty_weight=500.0)
            if _is_valid(p, r) and np.sum(r) > best_score:
                best_score = np.sum(r)
                best_pos = p.copy()
                best_radii = r.copy()
        
        # Simulated annealing
        if time.time() < deadline - 12:
            p, r = _simulated_annealing(
                pos_init, rad_init, n_steps=3000, t0=0.03, t_end=0.0005,
                seed=idx * 100, deadline=deadline)
            if _is_valid(p, r) and np.sum(r) > best_score:
                best_score = np.sum(r)
                best_pos = p.copy()
                best_radii = r.copy()
    
    # Final polish on best
    if time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
                best_score = np.sum(radii_pp)
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: simple grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
