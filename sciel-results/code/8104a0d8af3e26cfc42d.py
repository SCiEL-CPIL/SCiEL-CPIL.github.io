"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
_START_TIME = time.time()
_DEADLINE = 115  # safety margin within 120s limit

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _check_validity(circles, tol=1e-9):
    """Check if circles are valid: inside unit square and disjoint."""
    if len(circles) != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized pairwise distance check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    # Check upper triangle only (excluding diagonal)
    iu = np.triu_indices(N, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True

def _get_max_radii(circles):
    """Compute maximum possible radius for each circle given positions."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(circles)
    
    # Distance to boundaries
    d_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Distance to other circles: for circle i, max r_i = min_j (dist_ij - r_j)
    max_r = d_bound.copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            max_r[i] = min(max_r[i], dist - r[j])
    
    return np.maximum(max_r, 0)

def _simulated_annealing(circles, n_iter=50000, T0=0.01, T1=0.0001, seed=42):
    """
    Simulated annealing to maximize sum of radii.
    At each step: pick a circle, perturb its position and radius,
    accept if valid and improves, or with probability exp(-delta/T) if it worsens.
    """
    if time.time() > _DEADLINE - 5:
        return circles
    
    rng = np.random.RandomState(seed)
    current = circles.copy()
    current_score = np.sum(current[:, 2])
    
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        # Temperature schedule
        T = T0 * (T1 / T0) ** (it / n_iter)
        
        # Pick random circle
        idx = rng.randint(0, n - 1)
        
        new_circles = current.copy()
        
        # Perturb position
        scale = 0.03 * (1 + 0.5 * np.cos(it / 1000))
        new_circles[idx, 0] += rng.gauss(0, scale)
        new_circles[idx, 1] += rng.gauss(0, scale)
        
        # Perturb radius (try to grow)
        growth_bias = 0.3
        new_circles[idx, 2] *= (1 + rng.gauss(growth_bias, 0.1))
        new_circles[idx, 2] = min(max(new_circles[idx, 2], 0.005), 0.5)
        
        # Clamp position to stay within bounds considering radius
        r_new = new_circles[idx, 2]
        new_circles[idx, 0] = min(max(new_circles[idx, 0], r_new), 1 - r_new)
        new_circles[idx, 1] = min(max(new_circles[idx, 1], r_new), 1 - r_new)
        
        # Check validity
        if not _check_validity(new_circles):
            continue
        
        new_score = np.sum(new_circles[:, 2])
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = new_circles
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best

def _expand_to_max(circles, iterations=100):
    """Iteratively expand each circle to its maximum possible radius."""
    current = circles.copy()
    n = len(current)
    
    for it in range(iterations):
        if time.time() > _DEADLINE - 3:
            break
        
        improved = False
        for i in range(n):
            # Compute max radius for circle i given current positions and other radii
            x_i, y_i = current[i, 0], current[i, 1]
            max_r = min(x_i, 1 - x_i, y_i, 1 - y_i)
            
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt((x_i - current[j, 0])**2 + (y_i - current[j, 1])**2)
                max_r = min(max_r, dist - current[j, 2])
            
            max_r = max(max_r, 0.001)
            
            if max_r > current[i, 2] + 1e-6:
                # Try to set to max_r
                new_r = max_r
                # Verify validity
                test = current.copy()
                test[i, 2] = new_r
                if _check_validity(test):
                    current[i, 2] = new_r
                    improved = True
        
        if not improved:
            break
    
    return current

def _build_hex_initial(n_circles=N):
    """Build hexagonal packing initial guess."""
    best_circles = None
    best_score = -np.inf
    
    for spacing in np.linspace(0.15, 0.30, 20):
        sqrt3 = np.sqrt(3)
        positions = []
        for row in range(20):
            y = spacing * sqrt3 / 2 * row + spacing / 2
            if y > 1 - spacing / 2:
                break
            offset = spacing / 2 if row % 2 == 1 else 0
            for col in range(20):
                x = spacing * col + offset + spacing / 2
                if x > 1 - spacing / 2:
                    break
                positions.append((x, y))
        
        if len(positions) < n_circles:
            continue
        
        positions = positions[:n_circles]
        
        # Compute radii conservatively
        circles = []
        valid = True
        for k, (x, y) in enumerate(positions):
            d_bound = min(x, 1 - x, y, 1 - y)
            d_other = np.inf
            for m, (xm, ym) in enumerate(positions):
                if m == k:
                    continue
                d = np.sqrt((x - xm)**2 + (y - ym)**2)
                d_other = min(d_other, d)
            r = min(d_bound, d_other / 2) * 0.8
            if r < 0.005:
                valid = False
                break
            circles.append([x, y, r])
        
        if not valid:
            continue
        
        circles = np.array(circles)
        if _check_validity(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: grid
    side = int(np.ceil(np.sqrt(n_circles)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n_circles:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
    return np.array(cells[:n_circles])

def _build_random_initial(n_circles=N, seed=42):
    """Build random initial placement, made valid by shrinking."""
    rng = np.random.RandomState(seed)
    circles = []
    for _ in range(n_circles):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = rng.uniform(0.02, 0.06)
        circles.append([x, y, r])
    
    circles = np.array(circles)
    
    # Shrink to make valid
    for _ in range(50):
        if _check_validity(circles):
            break
        for i in range(n_circles):
            x, y, r = circles[i]
            max_r = min(x, 1 - x, y, 1 - y)
            for j in range(n_circles):
                if i == j:
                    continue
                dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                max_r = min(max_r, dist - circles[j, 2])
            circles[i, 2] = min(r, max(0.005, max_r * 0.9))
    
    if _check_validity(circles):
        return circles
    
    return _build_hex_initial(n_circles)

def _relax_positions(circles, n_iter=2000, lr=0.01, seed=42):
    """
    Relax positions to reduce overlaps using vectorized forces.
    """
    current = circles.copy()
    n = len(current)
    rng = np.random.RandomState(seed)
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        x = current[:, 0].copy()
        y = current[:, 1].copy()
        r = current[:, 2].copy()
        
        # Vectorized pairwise forces
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1.0)  # Avoid division by zero
        
        # Overlap amount
        overlap = r[:, None] + r[None, :] - dist
        overlap = np.maximum(overlap, 0)
        
        # Force direction (unit vector from j to i)
        inv_dist = 1.0 / dist
        fx = (overlap * dx * inv_dist)
        fy = (overlap * dy * inv_dist)
        
        # Sum forces on each circle
        fx_sum = np.sum(fx, axis=1)
        fy_sum = np.sum(fy, axis=1)
        
        # Boundary forces
        for i in range(n):
            if x[i] - r[i] < 0:
                fx_sum[i] += (r[i] - x[i]) * 5
            if x[i] + r[i] > 1:
                fx_sum[i] -= (x[i] + r[i] - 1) * 5
            if y[i] - r[i] < 0:
                fy_sum[i] += (r[i] - y[i]) * 5
            if y[i] + r[i] > 1:
                fy_sum[i] -= (y[i] + r[i] - 1) * 5
        
        # Update with adaptive learning rate
        lr_eff = lr * (1 - it / n_iter * 0.5)
        x += lr_eff * fx_sum
        y += lr_eff * fy_sum
        
        # Clamp
        for i in range(n):
            x[i] = min(max(x[i], r[i]), 1 - r[i])
            y[i] = min(max(y[i], r[i]), 1 - r[i])
        
        current[:, 0] = x
        current[:, 1] = y
    
    return current

def _two_phase_optimize(circles, seed=42):
    """Two-phase: relax positions, expand radii, then fine-tune with SA."""
    # Phase 1: Relax positions to reduce overlaps
    c1 = _relax_positions(circles, n_iter=2000, lr=0.01, seed=seed)
    if not _check_validity(c1):
        # If relaxation broke validity, fall back
        c1 = circles
    
    # Phase 2: Expand radii
    c2 = _expand_to_max(c1, iterations=50)
    
    # Phase 3: Simulated annealing for fine-tuning
    c3 = _simulated_annealing(c2, n_iter=30000, T0=0.005, T1=0.00005, seed=seed)
    
    # Phase 4: Final expansion
    c4 = _expand_to_max(c3, iterations=30)
    
    return c4

def construct():
    if time.time() > _DEADLINE:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    best_result = None
    best_value = -np.inf
    
    # Multiple initial configurations
    initializers = [
        ("hex", _build_hex_initial),
        ("rand1", lambda: _build_random_initial(N, seed=42)),
        ("rand2", lambda: _build_random_initial(N, seed=123)),
        ("rand3", lambda: _build_random_initial(N, seed=777)),
        ("rand4", lambda: _build_random_initial(N, seed=2024)),
    ]
    
    for name, init_fn in initializers:
        if time.time() > _DEADLINE - 15:
            break
        
        try:
            c0 = init_fn()
            if not _check_validity(c0):
                continue
            
            # Optimize with two-phase approach
            c_opt = _two_phase_optimize(c0, seed=hash(name) % 10000)
            
            if _check_validity(c_opt):
                val = np.sum(c_opt[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = c_opt.copy()
        except Exception:
            pass
    
    # If we have time, do another round of SA on the best result
    if best_result is not None and time.time() < _DEADLINE - 10:
        try:
            c_final = _simulated_annealing(best_result, n_iter=20000, T0=0.002, T1=0.00002, seed=9999)
            if _check_validity(c_final):
                val = np.sum(c_final[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = c_final.copy()
        except Exception:
            pass
    
    # Final expansion
    if best_result is not None:
        try:
            c_exp = _expand_to_max(best_result, iterations=20)
            if _check_validity(c_exp):
                val = np.sum(c_exp[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = c_exp.copy()
        except Exception:
            pass
    
    # Final validation
    if best_result is not None and _check_validity(best_result):
        return best_result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
