"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    # Vectorized overlap check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = r[:, None] + r[None, :]
    np.fill_diagonal(min_dist, 0)
    np.fill_diagonal(dist, np.inf)
    if np.any(dist < min_dist - 1e-9):
        return None
    return float(np.sum(r))


def _hex_initial(n=N):
    """Create a hexagonal packing initial configuration."""
    circles = np.zeros((n, 3))
    # Use hexagonal lattice
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    idx = 0
    for row in range(rows):
        cols = int(np.ceil(np.sqrt(n * 2 / 3))) if row % 2 == 0 else int(np.ceil(np.sqrt(n * 2 / 3))) - 1
        for col in range(cols):
            if idx >= n:
                break
            x = (col + 0.5 + 0.5 * (row % 2)) / cols
            y = (row + 0.5) / rows
            # Estimate radius based on spacing
            r = 0.35 / cols
            r = min(r, x, 1 - x, y, 1 - y)
            r = max(r, 0.01)
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    # Fill remaining
    if idx < n:
        rng = np.random.RandomState(42)
        for i in range(idx, n):
            r = 0.02 + 0.02 * rng.rand()
            circles[i, 0] = r + (1 - 2 * r) * rng.rand()
            circles[i, 1] = r + (1 - 2 * r) * rng.rand()
            circles[i, 2] = r
    return circles


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.45 / side
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.02 + 0.03 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _resolve_overlaps(circles, max_iter=1000, time_limit=5.0):
    """Resolve overlaps by moving circles apart."""
    start_time = time.time()
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-12:
                    if dist < 1e-12:
                        dx, dy = 1e-6, 1e-6
                        dist = np.sqrt(dx * dx + dy * dy)
                    overlap = min_dist - dist
                    # Move both circles apart
                    push = overlap / 2 + 1e-6
                    x[i] += push * dx / dist
                    y[i] += push * dy / dist
                    x[j] -= push * dx / dist
                    y[j] -= push * dy / dist
                    moved = True
        # Clamp to boundaries
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        if not moved:
            break
    
    return np.column_stack([x, y, r])


def _local_optimize(circles, time_limit=30.0, seed=0):
    """Local optimization using scipy SLSQP."""
    start_time = time.time()
    x0 = circles[:, 0].copy()
    y0 = circles[:, 1].copy()
    r0 = circles[:, 2].copy()
    n = len(r0)
    
    # Variables: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    v0 = np.concatenate([x0, y0, r0])
    
    def objective(v):
        if time.time() - start_time > time_limit:
            return 0
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        # Maximize sum of radii => minimize negative
        return -np.sum(r)
    
    # Constraints: boundary and non-overlap
    constraints = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    for i in range(n):
        # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n + i]})
        # x <= 1 - r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*n + i]})
        # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n + i] - v[2*n + i]})
        # y <= 1 - r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[n + i] - v[2*n + i]})
        # r > 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n + i] - 0.001})
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[i] - v[j])**2 + (v[n+i] - v[n+j])**2) - (v[2*n+i] + v[2*n+j])
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
    for i in range(n):
        bounds.append((0.001, 0.999))  # y
    for i in range(n):
        bounds.append((0.001, 0.5))  # r
    
    try:
        result = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        
        if result.success or (result.fun < objective(v0)):
            v = result.x
            new_circles = np.column_stack([v[0:n], v[n:2*n], v[2*n:3*n]])
            # Validate
            score = _validate_and_score(new_circles)
            if score is not None:
                return new_circles, score
    except Exception:
        pass
    
    # Fallback: return original
    score = _validate_and_score(circles)
    return circles, score


def _gradient_descent(circles, max_iter=5000, lr_init=0.01, time_limit=30, seed=0):
    """Penalty-based gradient descent."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Boundary penalties
        for i in range(n):
            if x[i] < r[i]:
                gx[i] -= 50
                gr[i] += 50
            if 1 - x[i] < r[i]:
                gx[i] += 50
                gr[i] += 50
            if y[i] < r[i]:
                gy[i] -= 50
                gr[i] += 50
            if 1 - y[i] < r[i]:
                gy[i] += 50
                gr[i] += 50
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dist = 1e-12
                    dx = 1e-12
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    gx[i] -= 30 * dx / dist
                    gy[i] -= 30 * dy / dist
                    gr[i] += 30
                    gx[j] += 30 * dx / dist
                    gy[j] += 30 * dy / dist
                    gr[j] += 30
        
        gr -= 1.0
        
        x -= lr * gx
        y -= lr * gy
        r -= lr * gr
        
        r = np.maximum(r, 0.001)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        for i in range(n):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] > max_r:
                r[i] = max(max_r, 0.001)
        
        lr *= 0.9999
        
        if it % 50 == 0:
            current = np.column_stack([x, y, r])
            score = _validate_and_score(current)
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_circles = current.copy()
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try multiple initializations
    inits = [
        (_hex_initial(N), "hex"),
        (_grid_initial(N), "grid"),
        (_random_initial(N, seed=42), "rand42"),
        (_random_initial(N, seed=123), "rand123"),
    ]
    
    best_circles = None
    best_score = None
    
    for init_circles, name in inits:
        if time.time() > deadline:
            break
        
        # Resolve overlaps first
        resolved = _resolve_overlaps(init_circles, max_iter=500, time_limit=3.0)
        
        # Try gradient descent
        gd_circles, gd_score = _gradient_descent(resolved, max_iter=3000, lr_init=0.008,
                                                  time_limit=20, seed=hash(name) % 1000)
        
        if gd_score is not None:
            if best_score is None or gd_score > best_score:
                best_score = gd_score
                best_circles = gd_circles
        
        # Try SLSQP local optimization
        remaining = deadline - time.time()
        if remaining > 10:
            slsqp_circles, slsqp_score = _local_optimize(gd_circles, time_limit=min(remaining - 5, 25))
            if slsqp_score is not None:
                if best_score is None or slsqp_score > best_score:
                    best_score = slsqp_score
                    best_circles = slsqp_circles
        
        # One more gradient descent pass
        remaining = deadline - time.time()
        if remaining > 5:
            gd2_circles, gd2_score = _gradient_descent(best_circles, max_iter=2000, lr_init=0.003,
                                                        time_limit=min(remaining - 2, 10), seed=999)
            if gd2_score is not None and gd2_score > best_score:
                best_score = gd2_score
                best_circles = gd2_circles
    
    # Final validation
    if best_circles is not None:
        final_score = _validate_and_score(best_circles)
        if final_score is not None:
            return best_circles
    
    # Fallback
    fallback = _grid_initial(N)
    fallback = _resolve_overlaps(fallback, max_iter=200, time_limit=2.0)
    return fallback

# EVOLVE-BLOCK-END

