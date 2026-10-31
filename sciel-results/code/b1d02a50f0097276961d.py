"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _slsqp_polish(pos, radii, deadline, maxiter=500):
    """
    SLSQP polish with explicit hard constraints for non-overlap and boundary.
    """
    x = np.concatenate([pos.flatten(), radii])
    n = len(radii)
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.9999))  # x
        bounds.append((1e-4, 0.9999))  # y
    for i in range(n):
        bounds.append((1e-4, 0.49))  # r
    
    # Inequality constraints: g(x) >= 0
    # Boundary constraints
    def _boundary_constraints(x):
        pos_c = x[:2*n].reshape(n, 2)
        r_c = x[2*n:]
        cons = []
        for i in range(n):
            cons.append(pos_c[i, 0] - r_c[i])  # x - r >= 0
            cons.append((1.0 - pos_c[i, 0]) - r_c[i])  # 1-x-r >= 0
            cons.append(pos_c[i, 1] - r_c[i])  # y - r >= 0
            cons.append((1.0 - pos_c[i, 1]) - r_c[i])  # 1-y-r >= 0
        return np.array(cons)
    
    # Overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    def _overlap_constraints(x):
        pos_c = x[:2*n].reshape(n, 2)
        r_c = x[2*n:]
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                dx = pos_c[i, 0] - pos_c[j, 0]
                dy = pos_c[i, 1] - pos_c[j, 1]
                dist_sq = dx*dx + dy*dy
                sum_r = r_c[i] + r_c[j]
                cons.append(dist_sq - sum_r*sum_r)
        return np.array(cons)
    
    constraints = [
        {'type': 'ineq', 'fun': _boundary_constraints},
        {'type': 'ineq', 'fun': _overlap_constraints}
    ]
    
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success or result.fun < _objective(x):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*n].reshape(n, 2)
    radii_out = np.maximum(x[2*n:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)
    
    return pos_out, radii_out

def _slsqp_radius_growth(pos, radii, deadline, maxiter=200):
    """
    SLSQP pass that fixes positions (with tiny tolerance) and maximizes radii.
    This squeezes additional sum-of-radii by growing circles into available gaps.
    """
    x = np.concatenate([pos.flatten(), radii])
    n = len(radii)
    
    # Bounds: positions fixed (tiny tolerance), radii can grow
    bounds = []
    for i in range(n):
        bounds.append((pos[i, 0] - 1e-6, pos[i, 0] + 1e-6))  # x fixed
        bounds.append((pos[i, 1] - 1e-6, pos[i, 1] + 1e-6))  # y fixed
    for i in range(n):
        bounds.append((radii[i] * 0.9999, 0.49))  # r can grow
    
    # Inequality constraints: g(x) >= 0
    def _boundary_constraints(x):
        pos_c = x[:2*n].reshape(n, 2)
        r_c = x[2*n:]
        cons = []
        for i in range(n):
            cons.append(pos_c[i, 0] - r_c[i])  # x - r >= 0
            cons.append((1.0 - pos_c[i, 0]) - r_c[i])  # 1-x-r >= 0
            cons.append(pos_c[i, 1] - r_c[i])  # y - r >= 0
            cons.append((1.0 - pos_c[i, 1]) - r_c[i])  # 1-y-r >= 0
        return np.array(cons)
    
    def _overlap_constraints(x):
        pos_c = x[:2*n].reshape(n, 2)
        r_c = x[2*n:]
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                dx = pos_c[i, 0] - pos_c[j, 0]
                dy = pos_c[i, 1] - pos_c[j, 1]
                dist_sq = dx*dx + dy*dy
                sum_r = r_c[i] + r_c[j]
                cons.append(dist_sq - sum_r*sum_r)
        return np.array(cons)
    
    constraints = [
        {'type': 'ineq', 'fun': _boundary_constraints},
        {'type': 'ineq', 'fun': _overlap_constraints}
    ]
    
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success or result.fun < _objective(x):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*n].reshape(n, 2)
    radii_out = np.maximum(x[2*n:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)
    
    return pos_out, radii_out

def _try_radius_expansion(pos, radii, max_iterations=200):
    """Try to expand each circle's radius while maintaining validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Max radius from walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            
            # Max radius from other circles
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])
            
            # Try to expand
            new_r = max_r_circ * 0.99999
            if new_r > r * 1.0001:
                # Test validity
                radii[i] = new_r
                if _is_valid(pos, radii):
                    improved = True
                else:
                    radii[i] = r
        
        if not improved:
            break
    
    return pos, radii

def _perturbation_local_search(pos, radii, n_perturbations=500, max_step=0.02, seed=0, deadline=None):
    """Local search with random perturbations."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for p in range(n_perturbations):
        if deadline is not None and time.time() > deadline - 2:
            break
            
        # Pick 1-3 circles to perturb
        n_pert = rng.randint(1, 4)
        indices = rng.choice(n, size=n_pert, replace=False)
        
        old_pos = pos[indices].copy()
        old_radii = radii[indices].copy()
        
        for idx in indices:
            dx = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dy = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dr = rng.normal(0, max_step * 0.3)
            
            new_x = pos[idx, 0] + dx
            new_y = pos[idx, 1] + dy
            new_r = radii[idx] + dr
            
            new_x = np.clip(new_x, 0.005, 0.995)
            new_y = np.clip(new_y, 0.005, 0.995)
            new_r = np.clip(new_r, 0.005, 0.4)
            
            pos[idx, 0] = new_x
            pos[idx, 1] = new_y
            radii[idx] = new_r
        
        # Check validity and score
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                # Update working copy
                pos = best_pos.copy()
                radii = best_radii.copy()
        else:
            # Revert
            pos[indices] = old_pos
            radii[indices] = old_radii
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with multi-stage local search, prioritizing SLSQP."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
    
    # Stage 1: SLSQP with explicit hard constraints (Primary)
    if time.time() < deadline - 30:
        pos_out, radii_out = _slsqp_polish(pos, radii, deadline, maxiter=500)
        pos = pos_out
        radii = radii_out
        x = np.concatenate([pos.flatten(), radii])
        
        # Radius Growth SLSQP
        if time.time() < deadline - 25:
            pos_out, radii_out = _slsqp_radius_growth(pos, radii, deadline, maxiter=200)
            pos = pos_out
            radii = radii_out
            x = np.concatenate([pos.flatten(), radii])
            
    # Stage 2: L-BFGS-B with high penalty on smooth objective (Secondary, for escape)
    if time.time() < deadline - 20:
        try:
            result = minimize(_objective_smooth, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0, 40.0),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective_smooth(x, 5000.0, 40.0) - 1e-12:
                x = result.x
        except Exception:
            pass
        
        pos_out = x[:2*N].reshape(N, 2)
        radii_out = np.maximum(x[2*N:], 1e-5)
        pos_out, radii_out = _post_process(pos_out, radii_out)
        
        # Re-polish with SLSQP after L-BFGS-B
        if time.time() < deadline - 15:
            pos_out, radii_out = _slsqp_polish(pos_out, radii_out, deadline, maxiter=300)
            pos = pos_out
            radii = radii_out
            x = np.concatenate([pos.flatten(), radii])
            
    # Stage 3: Perturbation local search to find better local optima
    if time.time() < deadline - 10:
        pos_out, radii_out = _perturbation_local_search(
            pos, radii, 
            n_perturbations=300, max_step=0.015, seed=42,
            deadline=deadline
        )
        pos = pos_out
        radii = radii_out
        x = np.concatenate([pos.flatten(), radii])
        
        # Final SLSQP polish
        if time.time() < deadline - 5:
            pos_out, radii_out = _slsqp_polish(pos, radii, deadline, maxiter=400)
            pos = pos_out
            radii = radii_out
            x = np.concatenate([pos.flatten(), radii])
            
    # Stage 4: Final radius expansion and growth
    if time.time() < deadline - 3:
        pos_out, radii_out = _try_radius_expansion(pos, radii, max_iterations=150)
        pos = pos_out
        radii = radii_out
        
        if time.time() < deadline - 2:
            pos_out, radii_out = _slsqp_radius_growth(pos, radii, deadline, maxiter=100)
            pos = pos_out
            radii = radii_out
            
    # Final post-process and validation
    pos_out, radii_out = _post_process(pos, radii)
    
    return pos_out, radii_out

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
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [50.0, 500.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Coarse L-BFGS-B to get close to feasible
        for p_w in penalty_stages:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
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
        
        # Stage 2: SLSQP Polish (Primary Refinement)
        if time.time() < deadline - 10:
            pos_slsqp, rad_slsqp = _slsqp_polish(current_x[:2*N].reshape(N, 2), np.maximum(current_x[2*N:], 1e-4), deadline, maxiter=300)
            pos_pp, radii_pp = _post_process(pos_slsqp, rad_slsqp)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
    # Final Polish on the best candidate found
    if time.time() < deadline - 5:
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
    
    # Extra SLSQP polish if time permits
    if time.time() < deadline - 3:
        try:
            pos_extra, rad_extra = _slsqp_polish(best_pos, best_radii, deadline, maxiter=200)
            pos_pp, radii_pp = _post_process(pos_extra, rad_extra)
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
