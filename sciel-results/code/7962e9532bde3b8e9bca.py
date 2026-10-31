"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth differentiable objective with soft-plus penalties, optimized by Simulated Annealing,
followed by hard-constraint SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
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

def _sum_radii(pos, radii):
    return np.sum(radii)

def _init_hexagonal():
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
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
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

def _init_grid():
    """Grid initialization with corner bias."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.09
                k += 1
    if k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
        k += 1
    return pos, radii

def _init_corners():
    """Corner-biased initialization."""
    rng = np.random.RandomState(42)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.17
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
                max_r2 = min(x2, 1 - x2, y2, 1 - y2)
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

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, radii - (1.0 - pos[:, 0]))
    bv += np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, radii - (1.0 - pos[:, 1]))
    bv = np.sum(_softplus(bv, beta))
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = np.maximum(0.0, sum_r - dist)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    ov = np.sum(_softplus(overlap[mask], beta))
    
    return -np.sum(radii) + penalty_weight * (bv + ov)

def _simulated_annealing(pos_init, radii_init, deadline, seed=0):
    """
    Simulated Annealing on the smooth objective.
    """
    rng = np.random.RandomState(seed)
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    
    T = 0.05
    cooling = 0.995
    best_x = x0.copy()
    best_score = _objective(x0)
    current_x = x0.copy()
    current_score = best_score
    
    step = 0
    while time.time() < deadline - 5:
        if step % 100 == 0 and T < 1e-4:
            break
        
        # Generate candidate
        x_new = current_x + rng.normal(0, T, x0.shape)
        # Keep radii positive
        x_new[2*N:] = np.maximum(x_new[2*N:], 1e-5)
        
        score_new = _objective(x_new)
        
        if score_new < current_score or rng.random() < np.exp((current_score - score_new) / T):
            current_x = x_new
            current_score = score_new
            if score_new < best_score:
                best_score = score_new
                best_x = x_new.copy()
        
        T *= cooling
        step += 1
    
    pos_out = best_x[:2*N].reshape(N, 2)
    radii_out = np.maximum(best_x[2*N:], 1e-5)
    return pos_out, radii_out

def _slsqp_polish(pos, radii, deadline):
    """
    SLSQP with hard inequality constraints for non-overlap and boundaries.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.4))
    
    constraints = []
    
    for i in range(N):
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if result.success or result.fun < objective(x0):
            x = result.x
            pos_out = x[:2*N].reshape(N, 2)
            radii_out = np.maximum(x[2*N:], 1e-4)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    pos_grid, rad_grid = _init_grid()
    inits.append((pos_grid, rad_grid))
    pos_cor, rad_cor = _init_corners()
    inits.append((pos_cor, rad_cor))
    pos_r1, rad_r1 = _init_random(100)
    inits.append((pos_r1, rad_r1))
    pos_r2, rad_r2 = _init_random(200)
    inits.append((pos_r2, rad_r2))
    
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = _sum_radii(best_pos, best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Simulated Annealing on smooth objective
        pos_sa, rad_sa = _simulated_annealing(pos_init, rad_init, deadline, seed=idx*100+1)
        pos_pp, rad_pp = _post_process(pos_sa, rad_sa)
        score = _sum_radii(pos_pp, rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
        
        # Phase 2: SLSQP polish
        if time.time() < deadline - 10:
            pos_sl, rad_sl = _slsqp_polish(pos_pp, rad_pp, deadline)
            pos_pp2, rad_pp2 = _post_process(pos_sl, rad_sl)
            score2 = _sum_radii(pos_pp2, rad_pp2)
            if _is_valid(pos_pp2, rad_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = rad_pp2.copy()
    
    # Final SLSQP on best
    if time.time() < deadline - 8:
        pos_final, rad_final = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_final, rad_final)
        score = _sum_radii(pos_pp, rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
    
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
