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
    
    # Penalty for going outside square (smooth)
    penalty = 0.0
    for i in range(n):
        # Left boundary
        if pos[i, 0] < radii[i]:
            penalty += (radii[i] - pos[i, 0])**2
        # Right boundary
        if pos[i, 0] > 1.0 - radii[i]:
            penalty += (pos[i, 0] - (1.0 - radii[i]))**2
        # Bottom boundary
        if pos[i, 1] < radii[i]:
            penalty += (radii[i] - pos[i, 1])**2
        # Top boundary
        if pos[i, 1] > 1.0 - radii[i]:
            penalty += (pos[i, 1] - (1.0 - radii[i]))**2
    
    # Penalty for overlapping (smooth)
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
    """Create a grid of n circles inside the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    """Create a hexagonal packing pattern."""
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
    
    # Inner ring: 8 circles
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i+1] = r_inner
    
    # Outer ring: 17 circles
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
        
        # Check boundaries
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.99
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        
        if valid:
            break
    
    # Final clamp
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
        bounds.append((radii[i], 1.0 - radii[i]))  # x
        bounds.append((radii[i], 1.0 - radii[i]))  # y
    for i in range(N):
        bounds.append((1e-4, 0.3))  # radius
    
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
    deadline = start_time + 140.0  # 10s safety margin
    
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
    
    # Generate initial configurations
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    
    rng = np.random.RandomState(42)
    for seed in range(5):
        configs.append(_make_random(N, seed=seed + 100))
    
    # Phase 1: L-BFGS-B optimization from each config
    for pos, radii in configs:
        if time.time() > deadline - 30:
            break
        
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
            bounds.append((0.01, 0.99))  # y
        for i in range(N):
            bounds.append((0.01, 0.3))  # radius
        
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement on best found
    if best_pos is not None and time.time() < deadline - 20:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result_nm = minimize(lambda x: _objective(x, deadline), x0, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'maxfev': 15000, 'xatol': 1e-8, 'fatol': 1e-8})
            pos_opt = result_nm.x[:2*N].reshape(N, 2)
            radii_opt = result_nm.x[2*N:]
            update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global exploration
    if time.time() < deadline - 40:
        try:
            def de_objective(x):
                return _objective(x, deadline)
            
            bounds_de = []
            for i in range(N):
                bounds_de.append((0.05, 0.95))  # x
                bounds_de.append((0.05, 0.95))  # y
            for i in range(N):
                bounds_de.append((0.02, 0.25))  # radius
            
            de_result = differential_evolution(de_objective, bounds_de, maxiter=80,
                                               popsize=15, seed=42, tol=1e-8,
                                               polish=False, mutation=(0.3, 1.0),
                                               recombination=0.7)
            
            if de_result.x is not None:
                pos_opt = de_result.x[:2*N].reshape(N, 2)
                radii_opt = de_result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 4: Polish the best result
    if best_pos is not None and time.time() < deadline - 5:
        best_pos, best_radii = _polish(best_pos, best_radii, deadline)
    
    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
