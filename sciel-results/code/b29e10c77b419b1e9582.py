"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

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


def _smooth_penalty(pos, radii, beta=80.0):
    """Compute smooth penalty for overlaps and boundary violations."""
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        for coord in range(2):
            val = pos[i, coord]
            r = radii[i]
            viol = r - val
            if viol > 0:
                penalty += viol * viol
            viol = val - (1.0 - r)
            if viol > 0:
                penalty += viol * viol
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            overlap = min_dist - dist
            if overlap > 0:
                penalty += overlap * overlap
    
    return penalty


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + penalty for violations."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    penalty = _smooth_penalty(pos, radii)
    
    return -np.sum(radii) + 500.0 * penalty


def _make_grid(n, seed=42):
    """5x5 grid with 26th circle in a gap."""
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    pos[25] = [0.5, 0.04]
    radii[25] = 0.035
    return pos, radii


def _make_hexagonal(n, seed=42):
    """Hexagonal packing pattern."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
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
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    """One center, rings around."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i+1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i+9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    """Random initial positions."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


def _make_dense_grid(n, seed=42):
    """Denser grid initialization with slight randomization."""
    rng = np.random.RandomState(seed)
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    for side in [4, 5, 6]:
        pos = np.zeros((n, 2))
        radii = np.zeros(n)
        cell = 1.0 / side
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                pos[count] = [(i + 0.5) * cell + rng.uniform(-0.01, 0.01),
                              (j + 0.5) * cell + rng.uniform(-0.01, 0.01)]
                radii[count] = cell * 0.45
                count += 1
        if count == n:
            if _is_valid(pos, radii, tol=1e-6):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
    
    if best_pos is None:
        return _make_grid(n, seed)
    return best_pos, best_radii


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(1e-8, max_r * 0.99)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        scale = 0.5
                    else:
                        scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(1e-8, radii[i] * scale)
                    radii[j] = max(1e-8, radii[j] * scale)
                    valid = False
        if valid:
            break
    
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-8, min(radii[i], max_r))
    
    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """High-precision polish using SLSQP with explicit constraints."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Build constraints
    constraints = []
    
    # Boundary constraints: pos[i] >= r[i], pos[i] <= 1 - r[i]
    for i in range(N):
        for coord in range(2):
            # pos >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, c=coord: x[2*i + c] - x[2*N + i],
                'jac': lambda x, i=i, c=coord: _boundary_jac(x, i, c, 0)
            })
            # pos <= 1 - r
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, c=coord: (1.0 - x[2*N + i]) - x[2*i + c],
                'jac': lambda x, i=i, c=coord: _boundary_jac(x, i, c, 1)
            })
    
    # Overlap constraints: dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: _overlap_val(x, i, j),
                'jac': lambda x, i=i, j=j: _overlap_jac(x, i, j)
            })
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.4))
    
    try:
        result = minimize(
            lambda x: _objective(x, deadline),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        if result.success or (result.fun < _objective(x0, deadline)):
            pos_opt = result.x[:2*N].reshape(N, 2)
            radii_opt = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _boundary_jac(x, i, coord, which):
    """Jacobian for boundary constraints."""
    jac = np.zeros(2*N + N)
    if which == 0:  # pos - r >= 0
        jac[2*i + coord] = 1.0
        jac[2*N + i] = -1.0
    else:  # (1-r) - pos >= 0
        jac[2*i + coord] = -1.0
        jac[2*N + i] = -1.0
    return jac


def _overlap_val(x, i, j):
    """Overlap constraint value: dist - (r_i + r_j) >= 0."""
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - (x[2*N + i] + x[2*N + j])


def _overlap_jac(x, i, j):
    """Jacobian for overlap constraint."""
    jac = np.zeros(2*N + N)
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist = np.sqrt(dx*dx + dy*dy)
    if dist > 1e-12:
        jac[2*i] = dx / dist
        jac[2*i+1] = dy / dist
        jac[2*j] = -dx / dist
        jac[2*j+1] = -dy / dist
    jac[2*N + i] = -1.0
    jac[2*N + j] = -1.0
    return jac


def _force_directed_anneal(pos, radii, deadline):
    """Force-directed annealing to improve packing."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    n = N
    temp = 0.01
    cooling = 0.995
    
    for step in range(500):
        if time.time() > deadline:
            break
        
        forces = np.zeros((n, 2))
        
        # Boundary forces
        for i in range(n):
            for coord in range(2):
                val = pos[i, coord]
                r = radii[i]
                if val < r:
                    forces[i, coord] += (r - val) * 10
                elif val > 1.0 - r:
                    forces[i, coord] -= (val - (1.0 - r)) * 10
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force_mag = overlap * 5
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Apply forces with annealing
        for i in range(n):
            pos[i, 0] += forces[i, 0] * temp
            pos[i, 1] += forces[i, 1] * temp
        
        # Clamp to square
        for i in range(n):
            for coord in range(2):
                pos[i, coord] = np.clip(pos[i, coord], 0.0, 1.0)
        
        # Cool down
        temp *= cooling
        
        # Periodically try to grow radii
        if step % 50 == 49:
            for i in range(n):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                if radii[i] < max_r * 0.95:
                    radii[i] = min(max_r * 0.95, radii[i] * 1.02)
    
    pos_pp, radii_pp = _post_process(pos, radii, deadline)
    if _is_valid(pos_pp, radii_pp):
        return pos_pp, radii_pp
    return pos, radii


def _growth_pass(pos, radii, deadline):
    """Try to grow radii incrementally while maintaining validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try to grow circle i
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] < max_r * 0.99:
                # Check if we can grow
                new_r = min(max_r * 0.99, radii[i] * 1.01)
                if new_r > radii[i] + 1e-8:
                    valid = True
                    for j in range(N):
                        if j == i:
                            continue
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < new_r + radii[j] - 1e-10:
                            valid = False
                            break
                    if valid:
                        radii[i] = new_r
                        improved = True
        if not improved:
            break
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Generate diverse initial configurations
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    configs.append(_make_dense_grid(N, seed=1))
    configs.append(_make_dense_grid(N, seed=2))
    for seed in range(6):
        configs.append(_make_random(N, seed=seed + 100))
    
    # Phase 1: L-BFGS-B from each config
    for idx, (pos, radii) in enumerate(configs):
        if time.time() > deadline - 50:
            break
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.3))
        
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            if result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 2: Force-directed annealing on best result
    if best_pos is not None and time.time() < deadline - 40:
        pos_fa, radii_fa = _force_directed_anneal(best_pos, best_radii, deadline)
        update_best(pos_fa, radii_fa)
    
    # Phase 3: SLSQP polish with explicit constraints
    if best_pos is not None and time.time() < deadline - 30:
        pos_p, radii_p = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            score = np.sum(radii_p)
            if score > best_score:
                best_score = score
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
    
    # Phase 4: Growth pass
    if best_pos is not None and time.time() < deadline - 20:
        pos_g, radii_g = _growth_pass(best_pos, best_radii, deadline)
        update_best(pos_g, radii_g)
    
    # Phase 5: Another SLSQP polish
    if best_pos is not None and time.time() < deadline - 15:
        pos_p, radii_p = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            score = np.sum(radii_p)
            if score > best_score:
                best_score = score
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
    
    # Phase 6: Final force-directed pass
    if best_pos is not None and time.time() < deadline - 10:
        pos_fa, radii_fa = _force_directed_anneal(best_pos, best_radii, deadline)
        update_best(pos_fa, radii_fa)
    
    # Phase 7: Final growth pass
    if best_pos is not None and time.time() < deadline - 5:
        pos_g, radii_g = _growth_pass(best_pos, best_radii, deadline)
        update_best(pos_g, radii_g)
    
    if best_pos is None:
        pos, radii = _make_grid(N)
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        return np.column_stack([pos_pp, radii_pp])
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
