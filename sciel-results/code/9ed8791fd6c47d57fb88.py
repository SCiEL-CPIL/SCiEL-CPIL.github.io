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

def _objective_epigraph(x, beta=30.0):
    """
    Epigraph formulation: introduce t >= |overlap violation| for each pair.
    x = [x1, y1, ..., xN, yN, r1, ..., rN, t1, ..., tM]
    where M = N*(N-1)/2 pairs.
    Minimize -sum(radii) + sum(t)
    Subject to: t_m >= softplus((r_i + r_j)^2 - dist^2)  (smooth)
    This is handled via penalty: minimize -sum(r) + sum(t) + penalty * sum(max(0, viol - t)^2)
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:3*N], 1e-6)
    t = x[3*N:]
    
    # Boundary penalty
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap: for each pair, violation = (r_i+r_j)^2 - dist^2
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = min_dist_sq - dist_sq  # positive when overlapping
    viol_masked = viol[mask]
    
    # Penalty: t should be >= viol (soft)
    # If t < viol, penalize
    pen = np.sum(np.maximum(0.0, viol_masked - t) ** 2)
    
    return -np.sum(radii) + np.sum(t) + 100.0 * boundary_pen + 1000.0 * pen

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

def _try_radius_expansion(pos, radii, max_iter=200, step_size=0.01):
    """
    Try to expand radii of circles that are not touching any constraint.
    Greedy local improvement with larger steps.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Check if circle i is touching a boundary
            touching_boundary = False
            if pos[i, 0] - radii[i] < 1e-4 or pos[i, 0] + radii[i] > 1.0 - 1e-4:
                touching_boundary = True
            if pos[i, 1] - radii[i] < 1e-4 or pos[i, 1] + radii[i] > 1.0 - 1e-4:
                touching_boundary = True
            
            # Check if circle i is touching any other circle
            touching_circle = False
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-4:
                    touching_circle = True
                    break
            
            # If not touching anything, try to expand
            if not touching_boundary and not touching_circle:
                max_expansion = 0.05
                
                # Check distance to boundaries
                max_expansion = min(max_expansion, pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, pos[i, 1] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 1] - radii[i])
                
                # Check distance to other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    space = dist - radii[i] - radii[j]
                    max_expansion = min(max_expansion, space)
                
                if max_expansion > 1e-5:
                    expansion = min(step_size, max_expansion * 0.8)
                    radii[i] += expansion
                    improved = True
        
        if not improved:
            break
    
    return pos, radii

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

def _slsqp_hard(pos, radii, deadline):
    """
    Optimize using SLSQP with explicit hard constraints (vectorized).
    """
    n = len(radii)
    x = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.99))
        bounds.append((1e-4, 0.99))
    for i in range(n):
        bounds.append((1e-4, 0.5))

    # Precompute pair indices
    pair_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def boundary_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        return np.concatenate([
            pos[:, 0] - radii,
            1.0 - pos[:, 0] - radii,
            pos[:, 1] - radii,
            1.0 - pos[:, 1] - radii
        ])

    def overlap_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = np.empty(len(pair_indices))
        for k, (i, j) in enumerate(pair_indices):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            cons[k] = dist_sq - min_dist*min_dist
        return cons

    def objective(x):
        radii = x[2*n:]
        return -np.sum(radii)

    def jac_objective(x):
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        return grad

    try:
        res = minimize(objective, x, method='SLSQP', bounds=bounds,
                       constraints=[
                           {'type': 'ineq', 'fun': boundary_constraints},
                           {'type': 'ineq', 'fun': overlap_constraints}
                       ],
                       jac=jac_objective,
                       options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.success or (hasattr(res, 'x') and res.fun < objective(x)):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_radii = np.maximum(res.x[2*n:], 1e-4)
            return new_pos, new_radii
    except Exception:
        pass
    
    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP hard constraints and local expansion."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Stage 1: SLSQP with hard constraints
    if time.time() < deadline - 15:
        pos, radii = _slsqp_hard(pos, radii, deadline)
        pos, radii = _post_process(pos, radii)
        
    # Stage 2: Local radius expansion
    if time.time() < deadline - 5:
        pos, radii = _try_radius_expansion(pos, radii, max_iter=200, step_size=0.01)
        pos, radii = _post_process(pos, radii)
        
    # Stage 3: Another SLSQP pass if time permits
    if time.time() < deadline - 5:
        pos, radii = _slsqp_hard(pos, radii, deadline)
        pos, radii = _try_radius_expansion(pos, radii, max_iter=100, step_size=0.005)
        pos, radii = _post_process(pos, radii)
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
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
        
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Quick penalty-based optimization (fewer iterations, focus on SLSQP)
        for p_w in [200.0, 1000.0]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
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
        
        # Stage 2: SLSQP Hard Constraints (primary optimization method)
        if time.time() < deadline - 20:
            try:
                pos_opt = current_x[:2*N].reshape(N, 2)
                rad_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_slq, rad_slq = _slsqp_hard(pos_opt, rad_opt, deadline)
                pos_exp, rad_exp = _try_radius_expansion(pos_slq, rad_slq, max_iter=100, step_size=0.01)
                pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish with multiple restarts
    if time.time() < deadline - 20:
        # Primary polish
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
        
        # Secondary polish from slightly perturbed initial
        if time.time() < deadline - 10:
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
        
        # Tertiary: Try a different perturbation strategy
        if time.time() < deadline - 5:
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
    
    # Final radius expansion attempt
    if time.time() < deadline - 1:
        pos_exp, rad_exp = _try_radius_expansion(best_pos, best_radii, max_iter=200, step_size=0.005)
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
