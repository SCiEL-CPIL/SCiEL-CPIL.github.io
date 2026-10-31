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


def _compute_penalty(pos, radii):
    """Compute smooth penalty for boundary and overlap violations."""
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        for coord in range(2):
            val = pos[i, coord]
            r = radii[i]
            viol_left = r - val
            if viol_left > 0:
                penalty += viol_left * viol_left
            viol_right = val - (1.0 - r)
            if viol_right > 0:
                penalty += viol_right * viol_right
    
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
    
    penalty = _compute_penalty(pos, radii)
    
    return -np.sum(radii) + 200.0 * penalty


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


def _coordinate_descent(pos, radii, deadline, max_iter=5000):
    """Coordinate descent: optimize one circle's position and radius at a time."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = N
    
    for outer_iter in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Optimize position of circle i given others fixed
            best_pos_i = pos[i].copy()
            best_r_i = radii[i]
            best_obj = 1e10
            
            # Try small perturbations in position
            for dx in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                for dy in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                    new_pos = pos.copy()
                    new_pos[i] = [pos[i, 0] + dx, pos[i, 1] + dy]
                    # Check bounds
                    if new_pos[i, 0] < radii[i] or new_pos[i, 0] > 1.0 - radii[i]:
                        continue
                    if new_pos[i, 1] < radii[i] or new_pos[i, 1] > 1.0 - radii[i]:
                        continue
                    
                    # Compute penalty only for circle i vs others
                    pen = 0.0
                    for j in range(n):
                        if j == i:
                            continue
                        dx2 = new_pos[i, 0] - pos[j, 0]
                        dy2 = new_pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        overlap = radii[i] + radii[j] - dist
                        if overlap > 0:
                            pen += overlap * overlap
                    
                    obj = -radii[i] + 200.0 * pen
                    if obj < best_obj:
                        best_obj = obj
                        best_pos_i = new_pos[i].copy()
                        improved = True
            
            # Try expanding radius
            for scale in [1.01, 1.02, 1.03, 1.05]:
                new_r = radii[i] * scale
                max_r = min(best_pos_i[0], 1.0 - best_pos_i[0], best_pos_i[1], 1.0 - best_pos_i[1])
                if new_r > max_r:
                    continue
                new_radii = radii.copy()
                new_radii[i] = new_r
                
                # Check overlaps with new radius
                pen = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    dx2 = best_pos_i[0] - pos[j, 0]
                    dy2 = best_pos_i[1] - pos[j, 1]
                    dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                    overlap = new_r + radii[j] - dist
                    if overlap > 0:
                        pen += overlap * overlap
                
                obj = -new_r + 200.0 * pen
                if obj < best_obj:
                    best_obj = obj
                    best_r_i = new_r
                    improved = True
            
            pos[i] = best_pos_i
            radii[i] = best_r_i
        
        if not improved:
            break
    
    return pos, radii


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


def _polish(pos, radii, deadline):
    """High-precision polish pass using L-BFGS-B with tight tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))
    
    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
        if result.fun < _objective(x0, deadline):
            pos_opt = result.x[:2*N].reshape(N, 2)
            radii_opt = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
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
        if time.time() > deadline - 60:
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
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            if result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 2: Coordinate descent on best result
    if best_pos is not None and time.time() < deadline - 40:
        pos_cd, radii_cd = _coordinate_descent(best_pos, best_radii, deadline, max_iter=3000)
        update_best(pos_cd, radii_cd)
    
    # Phase 3: Force-directed annealing on best result
    if best_pos is not None and time.time() < deadline - 30:
        pos_fa, radii_fa = _force_directed_anneal(best_pos, best_radii, deadline)
        update_best(pos_fa, radii_fa)
    
    # Phase 4: Polish the best result
    if best_pos is not None and time.time() < deadline - 20:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            score = np.sum(radii_p)
            if score > best_score:
                best_score = score
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
    
    # Phase 5: Another coordinate descent pass
    if best_pos is not None and time.time() < deadline - 15:
        pos_cd, radii_cd = _coordinate_descent(best_pos, best_radii, deadline, max_iter=2000)
        update_best(pos_cd, radii_cd)
    
    # Phase 6: Final polish
    if best_pos is not None and time.time() < deadline - 5:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            score = np.sum(radii_p)
            if score > best_score:
                best_score = score
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
    
    if best_pos is None:
        pos, radii = _make_grid(N)
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        return np.column_stack([pos_pp, radii_pp])
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
