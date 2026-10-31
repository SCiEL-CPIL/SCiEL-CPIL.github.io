import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # safety margin before 150s

def _is_valid(arr, tol=1e-6):
    """Check if arrangement is valid (all constraints satisfied)."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _objective(v, lam_b, lam_o):
    """v: [x(26), y(26), r(26)]. Minimize -sum(r) + penalties."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Objective: maximize sum(r) → minimize -sum(r)
    obj = -r.sum()
    
    # Boundary penalty (smooth)
    b = np.maximum(0, r - x)**2 + np.maximum(0, r - (1.0 - x))**2 + \
        np.maximum(0, r - y)**2 + np.maximum(0, r - (1.0 - y))**2
    obj += lam_b * b.sum()
    
    # Overlap penalty (smooth, vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rsum = r[:, None] + r[None, :]
    # Only upper triangle
    overlap = np.maximum(0, rsum - np.sqrt(np.maximum(0, dist2)))
    # Zero out diagonal
    np.fill_diagonal(overlap, 0.0)
    obj += lam_o * (overlap**2).sum()
    
    return obj

def _make_grid():
    """Grid initial guess - spread circles across the square."""
    # Use a 5x5 grid plus 1 extra = 26
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([(i+0.5)/5.0, (j+0.5)/5.0])
    # Add one more in the center area
    pts.append([0.5, 0.5])
    arr = np.array(pts[:N])
    r0 = 0.5/5.0 * 0.95
    radii = np.full(N, r0)
    return np.column_stack([arr[:,0], arr[:,1], radii])

def _make_hexagonal():
    """Hexagonal packing initial guess."""
    pts = []
    # Hexagonal pattern: rows offset
    for row in range(6):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.18
            if len(pts) < N:
                pts.append([x, y])
    arr = np.array(pts[:N])
    # Estimate radius
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((arr[i,0]-arr[j,0])**2 + (arr[i,1]-arr[j,1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.95
    for i in range(N):
        r0 = min(r0, arr[i,0], 1-arr[i,0], arr[i,1], 1-arr[i,1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:,0], arr[:,1], radii])

def _make_random(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    # Estimate radius from nearest neighbor
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    r = 0.4 * min_dists
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    r = np.maximum(r, 0.01)
    return np.column_stack([x, y, r])

def _optimize(arr_init, lam_b, lam_o, maxiter=300):
    """Run optimization from initial guess."""
    v0 = arr_init.flatten()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    res = minimize(
        _objective, v0,
        args=(lam_b, lam_o),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return res.x.reshape(N, 3)

def _post_process(arr):
    """Post-process: fix small violations and try to improve."""
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    
    # Fix boundary violations
    for i in range(N):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
    
    # Fix overlaps iteratively
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j]:
                    excess = r[i] + r[j] - d
                    # Reduce the larger circle more
                    if r[i] > r[j]:
                        r[i] = max(0, r[i] - excess * 0.6)
                        r[j] = max(0, r[j] - excess * 0.4)
                    else:
                        r[i] = max(0, r[i] - excess * 0.4)
                        r[j] = max(0, r[j] - excess * 0.6)
                    changed = True
        if not changed:
            break
    
    return np.column_stack([x, y, r])

def _polish(arr, time_budget=5.0):
    """High-precision polish pass using SLSQP with tight tolerances."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    v0 = np.concatenate([x, y, r])
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    start = time.time()
    try:
        res = minimize(
            _objective, v0,
            args=(1000.0, 10000.0),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16}
        )
        if res.success or res.fun < _objective(v0, 1000.0, 10000.0):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    return arr

def _force_directed_anneal(arr, time_budget=10.0):
    """Force-directed annealing to escape local minima."""
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    rng = np.random.RandomState(42)
    start = time.time()
    T = 0.1
    
    while time.time() - start < time_budget:
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 10
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] and d > 1e-10:
                    force = (r[i] + r[j] - d) * 5
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d
        
        # Apply forces with noise
        step = 0.01 * T
        x += step * fx + rng.normal(0, step*0.5, N)
        y += step * fy + rng.normal(0, step*0.5, N)
        
        # Clamp to bounds
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Cool down
        T *= 0.99
        
        # Check if valid
        if _is_valid(np.column_stack([x, y, r])):
            break
    
    return np.column_stack([x, y, r])

def construct():
    start_time = time.time()
    
    best_arr = None
    best_val = -np.inf
    
    # Generate initial guesses
    inits = []
    inits.append(_make_grid())
    inits.append(_make_hexagonal())
    
    # Several random inits
    for seed in [42, 123, 456, 789, 1024, 2048, 3333, 7777]:
        inits.append(_make_random(seed))
    
    # Multi-stage penalty optimization
    lambda_sets = [
        (10.0, 100.0),
        (50.0, 500.0),
        (100.0, 1000.0),
        (500.0, 5000.0),
    ]
    
    # Stage 1: Coarse optimization with low penalties
    stage1_results = []
    for lam_b, lam_o in lambda_sets[:2]:
        if time.time() - start_time > DEADLINE - 30:
            break
        for arr_init in inits:
            if time.time() - start_time > DEADLINE - 25:
                break
            try:
                arr = _optimize(arr_init, lam_b, lam_o, maxiter=200)
                arr = _post_process(arr)
                if _is_valid(arr):
                    val = arr[:, 2].sum()
                    if val > best_val:
                        best_val = val
                        best_arr = arr.copy()
                    stage1_results.append(arr)
            except Exception:
                continue
    
    # Stage 2: Fine optimization from best results
    if stage1_results:
        # Take top 3 from stage 1
        stage1_results.sort(key=lambda a: -a[:, 2].sum())
        top_results = stage1_results[:3]
        
        for lam_b, lam_o in lambda_sets[2:]:
            if time.time() - start_time > DEADLINE - 15:
                break
            for arr_init in top_results:
                if time.time() - start_time > DEADLINE - 10:
                    break
                try:
                    arr = _optimize(arr_init, lam_b, lam_o, maxiter=300)
                    arr = _post_process(arr)
                    if _is_valid(arr):
                        val = arr[:, 2].sum()
                        if val > best_val:
                            best_val = val
                            best_arr = arr.copy()
                except Exception:
                    continue
    
    # Stage 3: Polish the best result
    if best_arr is not None and time.time() - start_time < DEADLINE - 5:
        polished = _polish(best_arr, time_budget=5.0)
        polished = _post_process(polished)
        if _is_valid(polished):
            val = polished[:, 2].sum()
            if val > best_val:
                best_val = val
                best_arr = polished.copy()
    
    # Stage 4: Force-directed annealing if time permits
    if best_arr is not None and time.time() - start_time < DEADLINE - 3:
        annealed = _force_directed_anneal(best_arr, time_budget=min(5.0, DEADLINE - 3 - (time.time() - start_time)))
        annealed = _post_process(annealed)
        if _is_valid(annealed):
            val = annealed[:, 2].sum()
            if val > best_val:
                best_val = val
                best_arr = annealed.copy()
    
    # If no valid solution found, use grid as fallback
    if best_arr is None:
        best_arr = _make_grid()
        best_val = best_arr[:, 2].sum()
    
    # Final validation and cleanup
    if not _is_valid(best_arr):
        best_arr = _post_process(best_arr)
    
    if not _is_valid(best_arr):
        # Last resort: return grid
        best_arr = _make_grid()
    
    return best_arr
