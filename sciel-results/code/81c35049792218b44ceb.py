"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    penalty = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            penalty += (radii[i] - pos[i, 0])**2
        if pos[i, 0] > 1.0 - radii[i]:
            penalty += (pos[i, 0] - (1.0 - radii[i]))**2
        if pos[i, 1] < radii[i]:
            penalty += (radii[i] - pos[i, 1])**2
        if pos[i, 1] > 1.0 - radii[i]:
            penalty += (pos[i, 1] - (1.0 - radii[i]))**2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return -np.sum(radii) + 100.0 * penalty


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
    # 26th circle in a corner gap
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


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.99
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        if valid:
            break
    
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision polish pass with tighter tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((radii[i], 1.0 - radii[i]))
        bounds.append((radii[i], 1.0 - radii[i]))
    for i in range(N):
        bounds.append((1e-4, 0.3))
    
    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success:
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
    
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    for seed in range(5):
        configs.append(_make_random(N, seed=seed + 100))
    
    # Phase 1: L-BFGS-B from each config
    for pos, radii in configs:
        if time.time() > deadline - 30:
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
                              bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 2: Polish the best result
    if best_pos is not None and time.time() < deadline - 15:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            score = np.sum(radii_p)
            if score > best_score:
                best_score = score
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
    
    # Phase 3: Differential evolution for global exploration if time permits
    if time.time() < deadline - 25:
        try:
            bounds_de = []
            for i in range(N):
                bounds_de.append((0.01, 0.99))
                bounds_de.append((0.01, 0.99))
            for i in range(N):
                bounds_de.append((0.01, 0.25))
            
            result_de = differential_evolution(
                lambda x: _objective(x, deadline), bounds_de,
                seed=42, maxiter=30, popsize=10, tol=1e-6,
                mutation=(0.3, 1.0), recombination=0.8, init='sobol'
            )
            if result_de.success:
                pos_opt = result_de.x[:2*N].reshape(N, 2)
                radii_opt = result_de.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 4: Final polish
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
