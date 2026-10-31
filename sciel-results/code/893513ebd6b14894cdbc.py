import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 145.0  # Safety margin before 150s limit

def _objective(v, lam_b, lam_o):
    """
    Objective function: Minimize -sum(r) + penalties.
    v: flattened array [x(26), y(26), r(26)]
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Primary goal: maximize sum of radii
    obj = -r.sum()
    
    # Boundary penalties: circle must be inside [0,1]^2
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    b_x = np.maximum(0, r - x) + np.maximum(0, r - (1.0 - x))
    b_y = np.maximum(0, r - y) + np.maximum(0, r - (1.0 - y))
    obj += lam_b * (b_x.sum() + b_y.sum())
    
    # Overlap penalties: distance(i,j) >= r_i + r_j
    # Vectorized computation of pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    dist = np.sqrt(np.maximum(0, dist2))
    
    rsum = r[:, None] + r[None, :]
    
    # Overlap amount: max(0, r_i + r_j - dist_ij)
    # We square the overlap to make the penalty smoother and stronger for large violations
    overlap = np.maximum(0, rsum - dist)
    np.fill_diagonal(overlap, 0.0)
    
    obj += lam_o * (overlap**2).sum()
    
    return obj

def _is_valid(arr, tol=1e-5):
    """Check if the arrangement is strictly valid."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Check radii non-negative
    if np.any(r < 0):
        return False
        
    # Check boundaries
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _make_grid_initial():
    """Grid initial guess: 5x5 grid plus 1 extra."""
    side = 6
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    pts = pts[:N]
    arr = np.array(pts)
    # Initial radius: half the cell size, slightly reduced
    r0 = (0.5 / side) * 0.9
    # Ensure it fits in boundaries
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])

def _make_hex_initial():
    """Hexagonal-ish initial guess."""
    pts = []
    # 5 rows, 5 cols
    for row in range(5):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.2
            pts.append([x, y])
    pts.append([0.5, 0.5]) # 26th point
    
    # Scale to fit [0,1] with margin
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    scale = 0.8 / max(xmax - xmin, ymax - ymin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    for p in pts:
        p[0] = 0.5 + (p[0] - cx) * scale
        p[1] = 0.5 + (p[1] - cy) * scale
    
    arr = np.array(pts)
    # Estimate radius from min distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((arr[i, 0] - arr[j, 0])**2 + (arr[i, 1] - arr[j, 1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.9
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])

def _make_random_initial(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    
    # Calculate min distance to neighbors for radius estimation
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    
    r = 0.4 * min_dists
    # Bound by walls
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.01)
    
    return np.column_stack([x, y, r])

def _make_radial_initial():
    """
    Structured radial initialization: 1 center circle, 
    inner ring of 6, outer ring of 12, and 7 more scattered or in a 3rd ring.
    This aligns with the likely optimal topology (central + rings).
    """
    pts = []
    # Center
    r_c = 0.15
    pts.append([0.5, 0.5, r_c])
    
    # Inner ring: 6 circles
    r_i = 0.12
    d_i = 0.35
    for k in range(6):
        theta = k * np.pi / 3.0
        x = 0.5 + d_i * np.cos(theta)
        y = 0.5 + d_i * np.sin(theta)
        pts.append([x, y, r_i])
        
    # Outer ring: 12 circles
    r_o = 0.10
    d_o = 0.65
    for k in range(12):
        theta = k * np.pi / 6.0 + np.pi/12 # offset
        x = 0.5 + d_o * np.cos(theta)
        y = 0.5 + d_o * np.sin(theta)
        # Clamp to box
        x = np.clip(x, r_o, 1-r_o)
        y = np.clip(y, r_o, 1-r_o)
        pts.append([x, y, r_o])
        
    # 7 more circles: maybe a 3rd ring or corners
    # Let's put them in the corners and mid-edges
    # Corners: (0.1,0.1), (0.9,0.1), (0.1,0.9), (0.9,0.9)
    # Mid edges: (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)
    # We need 7. Let's take 4 corners and 3 mid edges.
    extra_pts = [
        [0.15, 0.15, 0.10],
        [0.85, 0.15, 0.10],
        [0.15, 0.85, 0.10],
        [0.85, 0.85, 0.10],
        [0.5, 0.15, 0.10],
        [0.5, 0.85, 0.10],
        [0.15, 0.5, 0.10]
    ]
    pts.extend(extra_pts)
    
    arr = np.array(pts[:N])
    # Shrink radii slightly to ensure initial validity or near-validity
    # Calculate min distance and shrink
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((arr[i,0]-arr[j,0])**2 + (arr[i,1]-arr[j,1])**2)
            min_dist = min(min_dist, d)
    # Scale radii by min_dist/2 * 0.95 to be safe
    # But keep relative sizes? Or just uniform shrink?
    # Let's just ensure they fit boundaries and are not overlapping too much
    # Simple repair:
    for i in range(N):
        d_wall = min(arr[i,0], arr[i,1], 1-arr[i,0], 1-arr[i,1])
        if arr[i,2] > d_wall:
            arr[i,2] = d_wall * 0.99
            
    # Overlap repair
    for _ in range(50):
        max_viol = 0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((arr[i,0]-arr[j,0])**2 + (arr[i,1]-arr[j,1])**2)
                if arr[i,2] + arr[j,2] > d:
                    viol = arr[i,2] + arr[j,2] - d
                    if viol > max_viol:
                        max_viol = viol
                    # Shrink both
                    arr[i,2] -= viol/2
                    arr[j,2] -= viol/2
        if max_viol < 1e-6:
            break
    arr[:,2] = np.maximum(arr[:,2], 0.01)
    return arr

def _repair(arr, max_iter=100):
    """
    Iteratively shrink radii to satisfy constraints if the optimizer failed.
    """
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    
    for _ in range(max_iter):
        # Check boundary violations
        max_viol = 0.0
        for i in range(N):
            d_wall = min(x[i], y[i], 1.0 - x[i], 1.0 - y[i])
            if r[i] > d_wall:
                max_viol = max(max_viol, r[i] - d_wall)
        
        # Check overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if r[i] + r[j] > d:
                    max_viol = max(max_viol, r[i] + r[j] - d)
        
        if max_viol < 1e-8:
            break
            
        # Shrink all radii by the max violation amount (conservative)
        shrink = max_viol / 2.0
        r -= shrink
        r = np.maximum(r, 0.0)
        
    return np.column_stack([x, y, r])

def _optimize_single(init_arr, start_lam_b, start_lam_o, time_limit):
    """
    Run iterative penalty optimization starting from init_arr.
    """
    v0 = init_arr.flatten()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    best_val = -np.inf
    best_v = v0.copy()
    
    # Iterative penalty increase
    lam_b = start_lam_b
    lam_o = start_lam_o
    
    # Multi-stage: 
    # Stage 1: Low penalty to find rough packing
    # Stage 2: Medium penalty to refine
    # Stage 3: High penalty to enforce constraints strictly
    
    stages = [
        (start_lam_b, start_lam_o, 50),
        (start_lam_b * 5, start_lam_o * 5, 100),
        (start_lam_b * 20, start_lam_o * 20, 200),
        (start_lam_b * 100, start_lam_o * 100, 300),
    ]
    
    for lam_b_stage, lam_o_stage, max_iter in stages:
        if time.time() > time_limit:
            break
            
        try:
            res = minimize(
                _objective, v0,
                args=(lam_b_stage, lam_o_stage),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'maxls': 50}
            )
            v0 = res.x
        except Exception:
            break
        
        # Check validity and update best
        arr = v0.reshape(N, 3)
        if _is_valid(arr):
            val = arr[:, 2].sum()
            if val > best_val:
                best_val = val
                best_v = v0.copy()
        else:
            # Even if invalid, keep the position but note it's not the best valid yet
            pass
            
    return best_v, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_arr = None
    best_val = -np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_make_grid_initial())
    inits.append(_make_hex_initial())
    inits.append(_make_radial_initial()) # New structural initialization
    
    # Random inits with different seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4000]
    for s in seeds:
        inits.append(_make_random_initial(s))
    
    # Base lambda values
    base_lam_b = 10.0
    base_lam_o = 10.0
    
    # Try each initialization
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            v_best, val = _optimize_single(init, base_lam_b, base_lam_o, deadline - 5)
        except Exception:
            continue
            
        if val > best_val:
            best_val = val
            best_arr = v_best.reshape(N, 3)
            
    # If no valid solution found from optimization, use a repaired version of the best attempt
    if best_arr is None:
        # Fallback: use grid and repair
        best_arr = _make_grid_initial()
        best_val = best_arr[:, 2].sum()
        
    # Final Polish: If the best solution is slightly invalid, try to repair it
    # Or, if it is valid, try one more high-penalty step to squeeze out more radius
    if _is_valid(best_arr):
        # Try to improve further with very high penalties on the valid solution
        if time.time() < deadline - 5:
            v0 = best_arr.flatten()
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            try:
                res = minimize(
                    _objective, v0,
                    args=(1000.0, 1000.0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-13}
                )
                arr_new = res.x.reshape(N, 3)
                if _is_valid(arr_new):
                    val_new = arr_new[:, 2].sum()
                    if val_new > best_val:
                        best_val = val_new
                        best_arr = arr_new
            except Exception:
                pass
    else:
        # Repair the invalid solution
        best_arr = _repair(best_arr)
        if _is_valid(best_arr):
            best_val = best_arr[:, 2].sum()
        else:
            # Last resort: return a known valid grid
            best_arr = _make_grid_initial()
            best_val = best_arr[:, 2].sum()
            
    return best_arr
