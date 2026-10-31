"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

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

def _post_process(pos, radii, max_iter=500):
    """Shrink radii to ensure strict validity, preserving total radius."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    # Shrink the smaller circle more
                    if radii[i] <= radii[j]:
                        k = dist / (sum_r + 1e-12) * 0.99999
                        radii[i] = max(radii[i] * k, 1e-6)
                    else:
                        k = dist / (sum_r + 1e-12) * 0.99999
                        radii[j] = max(radii[j] * k, 1e-6)
                    valid = False
        
        if valid:
            break
            
    return pos, radii

def _objective(x, penalty_weight=100.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
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
    """Smooth objective with softplus penalties."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    def softplus(v):
        return np.where(v > 20.0/beta, v, np.log1p(np.exp(beta * v)) / beta)
    
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)
    
    boundary_pen = np.sum(softplus(left_v)) + np.sum(softplus(right_v)) + \
                   np.sum(softplus(bottom_v)) + np.sum(softplus(top_v))
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(softplus(viol_raw[mask]))
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
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
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
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
        for _ in range(200):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.94
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

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _contact_graph(pos, radii, tol=1e-4):
    """Build contact graph: pairs of circles that are touching or nearly touching."""
    n = len(radii)
    contacts = set()
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] + tol:
                contacts.add((i, j))
    return contacts

def _qp_radius_optimization(pos, contacts, deadline):
    """
    Solve a QP to maximize sum of radii given fixed positions and contact graph.
    Constraints: r_i + r_j <= dist(i,j) for contacts, r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    """
    n = len(pos)
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        A_ub.append(np.eye(n)[i])
        b_ub.append(max_r)
    
    # Contact constraints: r_i + r_j <= dist(i,j)
    for (i, j) in contacts:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        row = np.zeros(n)
        row[i] = 1
        row[j] = 1
        A_ub.append(row)
        b_ub.append(dist)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0.001, 0.5)] * n
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return np.maximum(result.x, 0.001)
    except Exception:
        pass
    
    return None

def _iterative_qp_refinement(pos, radii, deadline, max_rounds=50):
    """
    Iteratively update contact graph and solve QP for radii,
    then use optimization to adjust positions.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for round_idx in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        # Build contact graph
        contacts = _contact_graph(pos, radii, tol=1e-3)
        
        # Solve QP for radii
        new_radii = _qp_radius_optimization(pos, contacts, deadline)
        if new_radii is not None:
            pos, radii = _post_process(pos, new_radii)
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        
        # Use SLSQP to optimize positions with current radii as target
        if time.time() > deadline - 5:
            break
        
        target_radii = radii.copy()
        
        def _pos_objective(x):
            p = x.reshape(N, 2)
            # Penalty for being too far from target radii
            # We want to maximize min possible radius, so we minimize violations
            total_viol = 0.0
            for i in range(N):
                # Boundary
                max_r = min(p[i,0], 1-p[i,0], p[i,1], 1-p[i,1])
                if target_radii[i] > max_r:
                    total_viol += (target_radii[i] - max_r)**2
            for i in range(N):
                for j in range(i+1, N):
                    dx = p[i,0] - p[j,0]
                    dy = p[i,1] - p[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    need = target_radii[i] + target_radii[j]
                    if dist < need:
                        total_viol += (need - dist)**2
            return total_viol
        
        x0 = pos.flatten()
        bounds = []
        for _ in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        
        try:
            result = minimize(_pos_objective, x0, method='SLSQP', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < 1e-10:
                new_pos = result.x.reshape(N, 2)
                # Re-solve QP with new positions
                contacts2 = _contact_graph(new_pos, target_radii, tol=1e-3)
                new_radii2 = _qp_radius_optimization(new_pos, contacts2, deadline)
                if new_radii2 is not None:
                    pos, radii = _post_process(new_pos, new_radii2)
                    if _is_valid(pos, radii):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_pos = pos.copy()
                            best_radii = radii.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0):
    """Simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 100.0)
    
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.015)
        dy = rng.normal(0, temp * 0.015)
        dr = rng.normal(0, temp * 0.008)
        
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.4)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 100.0)
        
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

def _try_radius_expansion(pos, radii, max_iter=100, step_size=0.005):
    """Greedy radius expansion for circles not touching constraints."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            touching_boundary = False
            if abs(pos[i, 0] - radii[i]) < 1e-5 or abs(pos[i, 0] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            if abs(pos[i, 1] - radii[i]) < 1e-5 or abs(pos[i, 1] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            
            touching_circle = False
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-5:
                    touching_circle = True
                    break
            
            if not touching_boundary and not touching_circle:
                max_expansion = 0.01
                max_expansion = min(max_expansion, pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, pos[i, 1] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 1] - radii[i])
                
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    space = dist - radii[i] - radii[j]
                    max_expansion = min(max_expansion, space)
                
                if max_expansion > 1e-6:
                    expansion = min(step_size, max_expansion * 0.5)
                    radii[i] += expansion
                    improved = True
        
        if not improved:
            break
    
    return pos, radii

def _polish(pos, radii, deadline):
    """Multi-stage polishing."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    # Stage 1: SLSQP
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          args=(5000.0,),
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 5000.0) - 1e-12:
            x = result.x
    except Exception:
        pass
    
    # Stage 2: L-BFGS-B
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0,),
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 10000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    # Stage 3: Smooth
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective_smooth, x, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0, 50.0),
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective_smooth(x, 10000.0, 50.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    # Stage 4: SLSQP final
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                              args=(20000.0,),
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
            if result.fun < _objective(x, 20000.0) - 1e-14:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    if time.time() < deadline - 2:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iter=100, step_size=0.002)
        pos_out, radii_out = _post_process(pos_out, radii_out)
        
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
        
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective with increasing penalty
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
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
        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
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
        
        # Stage 3: Simulated annealing
        if time.time() < deadline - 10:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=2000, t0=0.03, t_end=0.0005,
                    seed=idx * 100
                )
                pos_exp, rad_exp = _try_radius_expansion(pos_sa, rad_sa, max_iter=50, step_size=0.005)
                pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # QP-based refinement on best solution
    if time.time() < deadline - 20:
        try:
            pos_qp, rad_qp = _iterative_qp_refinement(best_pos, best_radii, deadline, max_rounds=30)
            if _is_valid(pos_qp, rad_qp):
                score = np.sum(rad_qp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_qp.copy()
                    best_radii = rad_qp.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 15:
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
        
        # Secondary polish from perturbed initial
        if time.time() < deadline - 8:
            rng = np.random.RandomState(42)
            pos_pert = best_pos + rng.normal(0, 0.005, (N, 2))
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            radii_pert = best_radii.copy()
            try:
                pos_pol2, rad_pol2 = _polish(pos_pert, radii_pert, deadline)
                pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
                score2 = np.sum(radii_pp2)
                if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                    best_score = score2
                    best_pos = pos_pp2.copy()
                    best_radii = radii_pp2.copy()
            except Exception:
                pass
        
        # Tertiary: Different perturbation
        if time.time() < deadline - 4:
            rng = np.random.RandomState(123)
            pos_pert3 = best_pos + rng.normal(0, 0.01, (N, 2))
            pos_pert3 = np.clip(pos_pert3, 0.01, 0.99)
            radii_pert3 = best_radii * 0.95
            try:
                pos_pol3, rad_pol3 = _polish(pos_pert3, radii_pert3, deadline)
                pos_pp3, radii_pp3 = _post_process(pos_pol3, rad_pol3)
                score3 = np.sum(radii_pp3)
                if _is_valid(pos_pp3, radii_pp3) and score3 > best_score:
                    best_score = score3
                    best_pos = pos_pp3.copy()
                    best_radii = radii_pp3.copy()
            except Exception:
                pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final radius expansion
    if time.time() < deadline - 1:
        pos_exp, rad_exp = _try_radius_expansion(best_pos, best_radii, max_iter=100, step_size=0.001)
        pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
            best_pos = pos_pp
            best_radii = radii_pp
            best_score = np.sum(radii_pp)
            
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
