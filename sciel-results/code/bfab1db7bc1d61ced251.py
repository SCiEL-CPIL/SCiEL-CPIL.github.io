"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus: log(1 + exp(beta*x)) / beta."""
    x = np.asarray(x, dtype=float)
    # For large positive x, softplus ≈ x
    # For large negative x, softplus ≈ 0
    out = np.zeros_like(x)
    pos = x > 0
    neg = ~pos
    out[pos] = x[pos] + np.log1p(np.exp(-beta * x[pos])) / beta
    out[neg] = np.log1p(np.exp(beta * x[neg])) / beta
    return out

def _soft_penalty(violations, beta=50.0):
    """
    Smooth penalty: sum(softplus(v * beta) * v) / beta
    Approximates sum(max(0, v)^2) when beta is large.
    """
    v = np.asarray(violations, dtype=float)
    v = np.clip(v, -20.0, 20.0)
    sp = _softplus(v, beta)
    return np.sum(sp * v)

def _objective(x, penalty_weight=1000.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty_weight * (boundary_violation + overlap_violation).
    x = [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    left_viol = radii - pos[:, 0]
    right_viol = pos[:, 0] - (1.0 - radii)
    bottom_viol = radii - pos[:, 1]
    top_viol = pos[:, 1] - (1.0 - radii)
    
    boundary_pen = (_soft_penalty(left_viol, beta) + _soft_penalty(right_viol, beta) +
                    _soft_penalty(bottom_viol, beta) + _soft_penalty(top_viol, beta))
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # violation = min_dist_sq - dist_sq (positive when overlapping)
    violation = min_dist_sq - dist_sq
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = _soft_penalty(violation[mask], beta)
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_epigraph(x, penalty_weight=1000.0):
    """
    Epigraph-style objective using auxiliary variable t.
    x = [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N, t]
    Minimize -sum(radii) + t + penalty_weight * (sum of (violation - t)_+^2)
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:3*N], 1e-6)
    t = x[3*N]
    
    # Boundary violations
    left_viol = radii - pos[:, 0]
    right_viol = pos[:, 0] - (1.0 - radii)
    bottom_viol = radii - pos[:, 1]
    top_viol = pos[:, 1] - (1.0 - radii)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    violation = min_dist_sq - dist_sq
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_viol = violation[mask]
    
    # Penalty: sum of (max(0, v - t))^2
    b_pen = 0.0
    for v_arr in [left_viol, right_viol, bottom_viol, top_viol]:
        excess = np.maximum(0.0, v_arr - t)
        b_pen += np.sum(excess ** 2)
    
    excess_o = np.maximum(0.0, overlap_viol - t)
    o_pen = np.sum(excess_o ** 2)
    
    return -np.sum(radii) + t + penalty_weight * (b_pen + o_pen)

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

def _post_process(pos, radii, n_iters=200):
    """
    Shrink radii slightly to ensure strict validity.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(n_iters):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 8000 random points for better coverage
        for _ in range(8000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
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

def _initial_corner_biased(seed=7):
    """
    Initialize with circles biased toward corners and edges.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place 4 circles near corners
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    # Place remaining circles with preference for edge regions
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
        
        max_r = min(x, 1.0-x, y, 1.0-y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0-x2, y2, 1.0-y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
    
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        for _ in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
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

def _force_directed_anneal(pos, radii, n_steps=5000, t0=0.05, t_end=0.001, seed=0):
    """
    Force-directed annealing.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        forces = np.zeros_like(pos)
        
        for i in range(n):
            r = radii[i]
            if pos[i, 0] < r:
                forces[i, 0] += (r - pos[i, 0]) * 10.0
            if pos[i, 0] > 1.0 - r:
                forces[i, 0] -= (pos[i, 0] - (1.0 - r)) * 10.0
            if pos[i, 1] < r:
                forces[i, 1] += (r - pos[i, 1]) * 10.0
            if pos[i, 1] > 1.0 - r:
                forces[i, 1] -= (pos[i, 1] - (1.0 - r)) * 10.0
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * 5.0
                    fy = (dy / dist) * overlap * 5.0
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        for i in range(n):
            pos[i, 0] += forces[i, 0] * 0.01 + rng.normal(0, temp * 0.01)
            pos[i, 1] += forces[i, 1] * 0.01 + rng.normal(0, temp * 0.01)
            
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r * 0.5, 1.0 - r * 0.5)
            pos[i, 1] = np.clip(pos[i, 1], r * 0.5, 1.0 - r * 0.5)
        
        if step % 100 == 50:
            for i in range(n):
                min_allowed = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
                for j in range(n):
                    if j != i:
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_allowed = min(min_allowed, dist - radii[j])
                if min_allowed > radii[i]:
                    radii[i] = min(min_allowed * 0.98, radii[i] * 1.02 + 0.001)
        
        if step % 200 == 0:
            valid = True
            for i in range(n):
                r = radii[i]
                if pos[i,0] - r < -1e-6 or pos[i,0] + r > 1.0 + 1e-6 or \
                   pos[i,1] - r < -1e-6 or pos[i,1] + r > 1.0 + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist_sq = dx*dx + dy*dy
                        min_d = radii[i] + radii[j]
                        if dist_sq < (min_d - 1e-6)**2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
    
    return best_pos, best_radii

def _high_precision_polish(pos, radii, time_limit=10.0):
    """
    High-precision polish using SLSQP with very tight tolerances.
    """
    start = time.time()
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    # Try SLSQP with high precision
    try:
        result = minimize(
            _objective, x0, method='SLSQP', bounds=bounds,
            args=(5000.0, 100.0),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if time.time() - start > time_limit:
            return pos, radii
    except Exception:
        pass
    
    pos_opt = result.x[:2*N].reshape(N, 2)
    radii_opt = np.maximum(result.x[2*N:], 1e-4)
    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
    
    if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii):
        return pos_pp, radii_pp
    return pos, radii

def _vectorized_refine(pos, radii, n_iters=100, time_limit=15.0):
    """
    Vectorized local refinement: iteratively adjust positions to maximize radii.
    """
    start = time.time()
    pos = pos.copy()
    radii = radii.copy()
    n = len(radii)
    
    for it in range(n_iters):
        if time.time() - start > time_limit:
            break
        
        # Compute max possible radius for each circle given current positions of others
        for i in range(n):
            # Boundary constraint
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            
            # Overlap constraint with all other circles
            dx = pos[i, 0] - pos[np.arange(n) != i, 0]
            dy = pos[i, 1] - pos[np.arange(n) != i, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            max_r = min(max_r, np.min(dists - radii[np.arange(n) != i]))
            
            # Grow radius if possible
            if max_r > radii[i] * 1.001:
                radii[i] = min(max_r * 0.999, radii[i] * 1.05)
        
        # After growing, do a few steps of position optimization to make room
        x0 = np.concatenate([pos.flatten(), radii])
        try:
            result = minimize(
                _objective, x0, method='L-BFGS-B',
                bounds=[(0.001, 0.999)] * (2*N) + [(0.001, 0.5)] * N,
                args=(2000.0, 80.0),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = np.maximum(result.x[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_new, radii_new)
            if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii):
                pos = pos_pp
                radii = radii_pp
        except Exception:
            pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Corner-biased
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    # 4. Random (2 variations)
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    # Multi-stage optimization with smooth penalties
    penalty_stages = [100.0, 500.0, 2000.0, 5000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective with increasing penalties
        for p_w in penalty_stages:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w, 50.0),
                                  options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-9})
                if result.fun < _objective(current_x, p_w, 50.0) - 1e-10:
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
        
        # Stage 2: Force-directed annealing from current state
        if time.time() < deadline - 20:
            try:
                pos_fd, rad_fd = _force_directed_anneal(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=4000, t0=0.03, t_end=0.0005,
                    seed=idx * 100
                )
                pos_pp, radii_pp = _post_process(pos_fd, rad_fd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 3: Vectorized refinement
        if time.time() < deadline - 15:
            try:
                pos_vr, rad_vr = _vectorized_refine(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_iters=50, time_limit=10.0
                )
                pos_pp, radii_pp = _post_process(pos_vr, rad_vr)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # If time allows, run Differential Evolution on the best result found so far
    if time.time() < deadline - 35:
        try:
            de_bounds = []
            for i in range(N):
                x, y = best_pos[i]
                r = best_radii[i]
                de_bounds.append((max(0.001, x - 0.15), min(0.999, x + 0.15)))
                de_bounds.append((max(0.001, y - 0.15), min(0.999, y + 0.15)))
            for i in range(N):
                r = best_radii[i]
                de_bounds.append((max(0.001, r * 0.6), min(0.5, r * 1.4)))
            
            result_de = differential_evolution(
                _objective, 
                de_bounds, 
                args=(1000.0, 80.0),
                maxiter=100, 
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=True
            )
            
            x_opt = result_de.x
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
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            pos_polished, radii_polished = _high_precision_polish(
                best_pos, best_radii, time_limit=10.0
            )
            score = np.sum(radii_polished)
            if _is_valid(pos_polished, radii_polished) and score > best_score:
                best_score = score
                best_pos = pos_polished.copy()
                best_radii = radii_polished.copy()
        except Exception:
            pass
    
    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
