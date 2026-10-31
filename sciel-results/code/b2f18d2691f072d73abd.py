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

def _post_process(pos, radii, iters=500):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(iters):
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

def _objective(x, penalty_weight=1000.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist - np.sqrt(dist_sq + 1e-12))
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_constrained(x):
    """
    For SLSQP: maximize sum(radii) subject to constraints.
    Returns -sum(radii) (to minimize).
    """
    radii = np.maximum(x[2*N:], 1e-6)
    return -np.sum(radii)

def _make_constraints():
    """Create constraint functions for SLSQP."""
    constraints = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i]})
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            def _non_overlap(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                return dist - x[2*N+i] - x[2*N+j]
            constraints.append({'type': 'ineq', 'fun': _non_overlap})
    
    return constraints

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.105
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
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _init_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.14, 0.14), (0.86, 0.14), (0.14, 0.86), (0.86, 0.86)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.19
    
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
        
        if max_r < 0.012:
            for attempt in range(300):
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
        radii[i] = max(max_r, 0.01) * 0.94
        
    return pos, radii

def _init_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.24
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(8000):
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
        
        if best_r < 0.003:
            best_x, best_y, best_r = 0.5, 0.5, 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.96
        
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.11, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=4000, t0=0.04, t_end=0.0003, seed=0):
    """
    Simulated annealing with incremental energy tracking.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Precompute pair indices for faster updates
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def _energy():
        e = -np.sum(radii)
        # Boundary
        for i in range(n):
            r = radii[i]
            x, y = pos[i]
            bv = max(0, r - x) + max(0, x - (1 - r)) + max(0, r - y) + max(0, y - (1 - r))
            e += 500.0 * bv * bv
        # Overlaps
        for i, j in pairs:
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = max(0.0, radii[i] + radii[j] - dist)
            e += 500.0 * viol * viol
        return e
    
    current_energy = _energy()
    
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        i = rng.randint(n)
        scale = 0.015 * (1.0 - frac * 0.7)
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        dr = rng.normal(0, scale * 0.6)
        
        new_x = np.clip(pos[i, 0] + dx, 0.005, 0.995)
        new_y = np.clip(pos[i, 1] + dy, 0.005, 0.995)
        new_r = np.clip(radii[i] + dr, 0.005, 0.4)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_energy = _energy()
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-10)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
            current_energy = _energy()
        
        if step % 200 == 0:
            pos_pp, radii_pp = _post_process(pos, radii, iters=100)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_polish(pos, radii, deadline):
    """SLSQP with hard constraints - allows radii to expand to true limit."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.45))
        
    constraints = _make_constraints()
    
    try:
        result = minimize(_objective_constrained, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success or result.fun < _objective_constrained(x):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-5)
    return pos_out, radii_out

def _lbfgs_polish(pos, radii, deadline):
    """L-BFGS-B with penalty objective."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.45))
        
    try:
        result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                          args=(5000.0,),
                          options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 5000.0) - 1e-12:
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-5)
    return pos_out, radii_out

def _expand_radii(pos, radii):
    """Try to expand each radius as much as possible given current positions."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(50):
        improved = False
        for i in range(N):
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(N):
                if j == i:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] * 1.001:
                radii[i] = max_r * 0.99999
                improved = True
        if not improved:
            break
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate initializations
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _init_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    pos_bf, rad_bf = _init_best_fit()
    inits.append((pos_bf, rad_bf))
    
    for i in range(3):
        pos_r, rad_r = _init_random(200 + i * 50)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        pos_cur = pos_init.copy()
        rad_cur = rad_init.copy()
        
        # Expand radii first
        pos_cur, rad_cur = _expand_radii(pos_cur, rad_cur)
        pos_pp, rad_pp = _post_process(pos_cur, rad_cur)
        if _is_valid(pos_pp, rad_pp):
            s = np.sum(rad_pp)
            if s > best_score:
                best_score = s
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()
        
        # L-BFGS-B polishing
        if time.time() < deadline - 15:
            pos_l, rad_l = _lbfgs_polish(pos_cur, rad_cur, deadline)
            pos_pp, rad_pp = _post_process(pos_l, rad_l)
            if _is_valid(pos_pp, rad_pp):
                s = np.sum(rad_pp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
        
        # SLSQP with hard constraints - key improvement
        if time.time() < deadline - 10:
            pos_s, rad_s = _slsqp_polish(pos_l, rad_l, deadline)
            pos_pp, rad_pp = _post_process(pos_s, rad_s, iters=200)
            if _is_valid(pos_pp, rad_pp):
                s = np.sum(rad_pp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
        
        # Simulated annealing
        if time.time() < deadline - 12:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    best_pos, best_radii,
                    n_steps=2500, t0=0.03, t_end=0.0002,
                    seed=idx * 137 + 1
                )
                pos_pp, rad_pp = _post_process(pos_sa, rad_sa)
                if _is_valid(pos_pp, rad_pp):
                    s = np.sum(rad_pp)
                    if s > best_score:
                        best_score = s
                        best_pos = pos_pp.copy()
                        best_radii = rad_pp.copy()
            except Exception:
                pass
    
    # Final round: SLSQP on best, then expand, then SLSQP again
    if time.time() < deadline - 8:
        pos_s2, rad_s2 = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_s2, rad_s2, iters=200)
        if _is_valid(pos_pp, rad_pp):
            s = np.sum(rad_pp)
            if s > best_score:
                best_score = s
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()
    
    # Expand radii on final best
    if time.time() < deadline - 3:
        pos_e, rad_e = _expand_radii(best_pos, best_radii)
        pos_pp, rad_pp = _post_process(pos_e, rad_e, iters=200)
        if _is_valid(pos_pp, rad_pp):
            s = np.sum(rad_pp)
            if s > best_score:
                best_score = s
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()
    
    # One more SLSQP after expansion
    if time.time() < deadline - 5:
        pos_s3, rad_s3 = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_s3, rad_s3, iters=200)
        if _is_valid(pos_pp, rad_pp):
            s = np.sum(rad_pp)
            if s > best_score:
                best_score = s
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii, iters=300)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: simple grid
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
        best_pos, best_radii = _post_process(pos, radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
