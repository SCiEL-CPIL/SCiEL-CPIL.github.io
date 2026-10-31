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


def _objective(x, lam=300.0):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-5)
    
    penalty = 0.0
    # Boundary penalties (smooth, cubic for better gradient near boundary)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-10)))
    
    return -np.sum(radii) + lam * penalty


def _make_hex_rows():
    """Generate hexagonal rows: alternating 5 and 4 circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.096
    y_spacing = r * np.sqrt(3)
    
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    
    idx = 0
    for row in range(5):
        if row % 2 == 0:
            count = 5
            x_start = x_start_5
        else:
            count = 4
            x_start = x_start_4
        for i in range(count):
            pos[idx, 0] = x_start + i * 2 * r
            pos[idx, 1] = y_start + r + row * y_spacing
            radii[idx] = r
            idx += 1
    
    # 26th circle: find best gap position
    best_score = -1
    best_pos = (0.5, 0.5)
    for cx in np.linspace(0.03, 0.97, 15):
        for cy in np.linspace(0.03, 0.97, 15):
            min_clearance = 1.0
            for i in range(25):
                dist = np.sqrt((cx - pos[i, 0])**2 + (cy - pos[i, 1])**2)
                clearance = dist - radii[i]
                min_clearance = min(min_clearance, clearance)
            min_clearance = min(min_clearance, cx, 1-cx, cy, 1-cy)
            if min_clearance > best_score:
                best_score = min_clearance
                best_pos = (cx, cy)
    
    pos[25, 0] = best_pos[0]
    pos[25, 1] = best_pos[1]
    radii[25] = max(best_score * 0.85, 0.015)
    
    return pos, radii


def _make_grid():
    """5x5 grid layout with 26th in a corner."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.097
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    pos[25, 0] = 0.08
    pos[25, 1] = 0.92
    radii[25] = 0.07
    return pos, radii


def _make_ring():
    """Central circle with surrounding ring."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.11
    
    ring_r = 0.075
    ring_dist = 0.11 + ring_r
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    corner_r = 0.055
    corners = [(0.07, 0.07), (0.93, 0.07), (0.07, 0.93), (0.93, 0.93),
               (0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5),
               (0.22, 0.22), (0.78, 0.22), (0.22, 0.78), (0.78, 0.78),
               (0.5, 0.3)]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
    return pos, radii


def _make_random(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.12, 0.88, size=(N, 2))
    radii = rng.uniform(0.05, 0.11, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-5)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-5)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-5)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-5)
                    radii[j] = max(radii[j] * scale, 1e-5)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _local_refine(pos, radii, deadline, lam=300.0):
    """Multi-method local refinement."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.3)] * N
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    methods = [
        ('L-BFGS-B', {'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8}),
        ('Powell', {'maxiter': 2000, 'ftol': 1e-13}),
        ('L-BFGS-B', {'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}),
    ]
    
    for method, opts in methods:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_objective, x0, method=method, bounds=bounds, options=opts, args=(lam,))
            if res.fun < _objective(x0, lam) - 1e-15:
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 3:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13}, args=(lam,))
            if res.fun < _objective(x0, lam) - 1e-15:
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _global_search(deadline):
    """Differential evolution global search."""
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.25)] * N
    de_deadline = deadline - 20
    
    def wrapper(x):
        return _objective(x, 200.0)
    
    try:
        res = differential_evolution(
            wrapper, bounds,
            maxiter=100, popsize=15,
            tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
            seed=7, polish=False, updating='deferred',
            callback=lambda xk, conv: (time.time() > de_deadline)
        )
        x_opt = res.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def construct():
    start = time.time()
    deadline = start + 145.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build diverse initial configurations
    configs = []
    p, r = _make_hex_rows()
    configs.append((p, r))
    p, r = _make_grid()
    configs.append((p, r))
    p, r = _make_ring()
    configs.append((p, r))
    for s in [1, 7, 42, 99, 200]:
        p, r = _make_random(s)
        configs.append((p, r))
    
    # Perturbed versions of structured inits
    rng = np.random.RandomState(123)
    for base_idx in [0, 1]:
        p, r = configs[base_idx]
        pp = p + rng.normal(0, 0.015, size=p.shape)
        pp = np.clip(pp, 0.03, 0.97)
        configs.append((pp, r.copy()))
    
    # Optimize from each config with time budget
    time_per_config = max(15, (deadline - 40) / len(configs))
    
    for idx, (pos, radii) in enumerate(configs):
        t_rem = deadline - time.time()
        if t_rem < 20:
            break
        alloc = min(time_per_config, t_rem - 15)
        if alloc < 8:
            continue
        
        pos_opt, radii_opt, score = _local_refine(pos, radii, start + time.time() - start + alloc, lam=300.0)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Global search if time allows
    if time.time() < deadline - 30 and best_score < 2.3:
        de_pos, de_radii, de_score = _global_search(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    # Fallback
    if best_pos is None or best_score < 0:
        p, r = _make_hex_rows()
        p, r = _post_process(p, r)
        best_pos = p
        best_radii = r
        best_score = np.sum(r)
    
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _local_refine(best_pos, best_radii, deadline, lam=150.0)
    
    # Ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
