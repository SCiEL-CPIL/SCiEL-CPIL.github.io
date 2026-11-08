"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
_START_TIME = time.time()
_DEADLINE = 145  # safety margin within 150s limit

# EVOLVE-BLOCK-START
import numpy as np
import random


def _check_validity(circles, tol=1e-9):
    """Check if circles are valid: inside unit square and disjoint."""
    if len(circles) != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized pairwise distance check
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _vectorized_check(circles, tol=1e-9):
    """Fast vectorized validity check."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized pairwise check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    # Only check upper triangle
    iu = np.triu_indices(N, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True


def _build_grid(n=N):
    """Build a grid initial configuration."""
    # 5x6 grid = 30 cells, use 26
    cols, rows = 5, 6
    circles = []
    for j in range(rows):
        for i in range(cols):
            if len(circles) >= n:
                break
            cx = (i + 0.5) / cols
            cy = (j + 0.5) / rows
            r = 0.5 / max(cols, rows) * 0.99
            circles.append([cx, cy, r])
    return np.array(circles[:n])


def _build_hex(n=N):
    """Build hexagonal initial configuration."""
    sqrt3 = np.sqrt(3)
    positions = []
    # Try to fit hexagonal packing
    for spacing in np.linspace(0.17, 0.25, 12):
        row_h = spacing * sqrt3 / 2
        for row in range(12):
            y = row_h * row + spacing * 0.5
            if y > 1 - spacing * 0.5:
                break
            offset = spacing * 0.5 if row % 2 == 1 else 0.0
            for col in range(12):
                x = spacing * col + offset + spacing * 0.5
                if x > 1 - spacing * 0.5:
                    break
                positions.append((x, y))
        if len(positions) >= n:
            positions = positions[:n]
            # Compute radii
            circles = []
            for k, (x, y) in enumerate(positions):
                d_bound = min(x, 1 - x, y, 1 - y)
                d_other = np.inf
                for m, (xm, ym) in enumerate(positions):
                    if m == k:
                        continue
                    d = np.sqrt((x - xm)**2 + (y - ym)**2)
                    d_other = min(d_other, d)
                r = min(d_bound, d_other / 2) * 0.95
                if r < 0.005:
                    r = 0.005
                circles.append([x, y, r])
            circles = np.array(circles)
            if _vectorized_check(circles):
                return circles
    # Fallback to grid
    return _build_grid(n)


def _relax_and_grow(circles, max_time=None):
    """
    Physics-based relaxation with simultaneous radius growth.
    Uses vectorized operations for speed.
    """
    if max_time is None:
        max_time = _DEADLINE - 5
    current = circles.copy()
    n = len(current)
    
    # Precompute indices for pairwise
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    for it in range(20000):
        if time.time() > max_time:
            break
        
        x = current[:, 0]
        y = current[:, 1]
        r = current[:, 2]
        
        # Compute pairwise distances
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(dx**2 + dy**2)
        desired = r[idx_i] + r[idx_j]
        
        # Forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Overlap forces (only where dist < desired)
        overlap = desired - dist
        mask = overlap > 0
        if np.any(mask):
            safe_dist = np.where(dist < 1e-10, 1e-10, dist)
            force = overlap[mask] / safe_dist[mask]
            fx[idx_i[mask]] += force * dx[mask]
            fy[idx_i[mask]] += force * dy[mask]
            fx[idx_j[mask]] -= force * dx[mask]
            fy[idx_j[mask]] -= force * dy[mask]
        
        # Boundary forces
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (r[i] - x[i]) * 5.0
            if x[i] + r[i] > 1:
                fx[i] -= (x[i] + r[i] - 1) * 5.0
            if y[i] - r[i] < 0:
                fy[i] += (r[i] - y[i]) * 5.0
            if y[i] + r[i] > 1:
                fy[i] -= (y[i] + r[i] - 1) * 5.0
        
        # Learning rate decay
        lr = 0.002 * (1.0 - it / 20000.0) + 0.0001
        
        # Update positions
        current[:, 0] += lr * fx
        current[:, 1] += lr * fy
        
        # Clamp to valid region
        for i in range(n):
            current[i, 0] = max(r[i] * 0.999, min(1 - r[i] * 0.999, current[i, 0]))
            current[i, 1] = max(r[i] * 0.999, min(1 - r[i] * 0.999, current[i, 1]))
        
        # Try to grow radii slightly
        if it % 50 == 0:
            grow = 0.0002 * (1.0 - it / 20000.0) + 0.00002
            new_r = current[:, 2] + grow
            # Check if growth is valid
            if _vectorized_check(np.column_stack([current[:, 0], current[:, 1], new_r])):
                current[:, 2] = new_r
            else:
                # Try smaller growth
                for frac in [0.5, 0.25, 0.1]:
                    trial_r = current[:, 2] + grow * frac
                    if _vectorized_check(np.column_stack([current[:, 0], current[:, 1], trial_r])):
                        current[:, 2] = trial_r
                        break
    
    return current


def _local_search(circles, n_iter=3000, seed=42):
    """Local search with position and radius perturbations."""
    if time.time() > _DEADLINE - 5:
        return circles
    
    current = circles.copy()
    current_score = np.sum(current[:, 2])
    best = current.copy()
    best_score = current_score
    rng = random.Random(seed)
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        idx = rng.randint(0, N - 1)
        new_c = current.copy()
        
        # Adaptive scale
        scale = 0.015 * (1.0 - it / float(n_iter)) + 0.003
        new_c[idx, 0] += rng.gauss(0, scale)
        new_c[idx, 1] += rng.gauss(0, scale)
        
        # Try to grow radius
        new_c[idx, 2] *= (1 + rng.gauss(0, 0.08))
        new_c[idx, 2] = min(max(new_c[idx, 2], 0.005), 0.49)
        
        if _vectorized_check(new_c):
            new_score = np.sum(new_c[:, 2])
            if new_score > current_score:
                current = new_c
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
    
    return best


def _final_growth(circles):
    """Aggressively try to grow all radii."""
    current = circles.copy()
    n = len(current)
    
    for outer in range(100):
        if time.time() > _DEADLINE - 3:
            break
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - 3:
                break
            # Try growing circle i
            for delta in [0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                new_r = current[i, 2] + delta
                if new_r > 0.49:
                    break
                trial = current.copy()
                trial[i, 2] = new_r
                if _vectorized_check(trial):
                    current[i, 2] = new_r
                    improved = True
                    break
        if not improved:
            break
    
    return current


def _polish_with_l_bfgs(circles, max_time=None):
    """
    Use scipy.optimize.minimize with L-BFGS-B for high-precision polish.
    Maximizes sum of radii by minimizing negative sum.
    Uses a penalty approach for constraints to make it compatible with L-BFGS-B.
    """
    from scipy.optimize import minimize
    
    if max_time is None:
        max_time = _DEADLINE - 5
    
    current = circles.copy()
    n = len(current)
    
    # Flatten to 3n variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = current.flatten()
    
    def objective(v):
        """Negative sum of radii plus penalty for constraint violations."""
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Base objective: minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        pen = 0.0
        for i in range(n):
            if x[i] - r[i] < 0:
                pen += (x[i] - r[i])**2
            if x[i] + r[i] > 1:
                pen += (x[i] + r[i] - 1)**2
            if y[i] - r[i] < 0:
                pen += (y[i] - r[i])**2
            if y[i] + r[i] > 1:
                pen += (y[i] + r[i] - 1)**2
        
        # Penalty for overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    pen += (min_d - d)**2
        
        # Small penalty for negative radii
        if np.any(r < 0):
            pen += np.sum(np.clip(-r, 0, None)**2)
        
        return obj + 100.0 * pen
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))   # x
        bounds.append((0, 1))   # y
        bounds.append((0.001, 0.49))  # r
    
    best_result = current.copy()
    best_val = np.sum(current[:, 2])
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if _vectorized_check(result.x.reshape(n, 3)):
            val = np.sum(result.x.reshape(n, 3)[:, 2])
            if val > best_val:
                best_val = val
                best_result = result.x.reshape(n, 3).copy()
    except Exception:
        pass
    
    return best_result


def construct():
    start = time.time()
    
    # Fallback
    def _fallback():
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.99) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    best_result = None
    best_value = -np.inf
    
    # Multiple initial configs
    inits = []
    g = _build_grid()
    if _vectorized_check(g):
        inits.append(g)
    h = _build_hex()
    if _vectorized_check(h):
        inits.append(h)
    
    # Random inits
    for seed in [1, 2, 3, 7, 42]:
        rng = np.random.RandomState(seed)
        c = []
        for _ in range(N):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = rng.uniform(0.02, 0.06)
            c.append([x, y, r])
        c = np.array(c)
        # Shrink to be valid
        for i in range(N):
            d_bound = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
            c[i,2] = min(c[i,2], d_bound * 0.9)
        if _vectorized_check(c):
            inits.append(c)
    
    if not inits:
        return _fallback()
    
    # Process each initial config
    for k, c0 in enumerate(inits):
        if time.time() > _DEADLINE - 25:
            break
        
        # Phase 1: Relax and grow
        c1 = _relax_and_grow(c0)
        if not _vectorized_check(c1):
            c1 = c0
        
        # Phase 2: Local search
        c2 = _local_search(c1, n_iter=2000, seed=100 + k)
        if not _vectorized_check(c2):
            c2 = c1
        
        # Phase 3: More relaxation
        c3 = _relax_and_grow(c2)
        if not _vectorized_check(c3):
            c3 = c2
        
        # Phase 4: Final local search
        c4 = _local_search(c3, n_iter=1500, seed=500 + k)
        if not _vectorized_check(c4):
            c4 = c3
        
        # Phase 5: Final growth
        c5 = _final_growth(c4)
        if not _vectorized_check(c5):
            c5 = c4
        
        if _vectorized_check(c5):
            val = np.sum(c5[:, 2])
            if val > best_value:
                best_value = val
                best_result = c5.copy()
    
    # Phase 6: L-BFGS-B polish on the best result
    if best_result is not None and time.time() < _DEADLINE - 8:
        polished = _polish_with_l_bfgs(best_result)
        if _vectorized_check(polished):
            val = np.sum(polished[:, 2])
            if val > best_value:
                best_value = val
                best_result = polished.copy()
    
    # Phase 7: Another round of local search + growth after polish
    if best_result is not None and time.time() < _DEADLINE - 5:
        c7 = _local_search(best_result, n_iter=1000, seed=999)
        if _vectorized_check(c7):
            c7 = _final_growth(c7)
            if _vectorized_check(c7):
                val = np.sum(c7[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = c7.copy()
    
    if best_result is not None and _check_validity(best_result):
        return best_result
    
    return _fallback()
# EVOLVE-BLOCK-END
