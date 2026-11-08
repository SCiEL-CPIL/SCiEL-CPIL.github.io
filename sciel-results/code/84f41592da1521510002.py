import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _voronoi_inradii(x, y, W, H):
    """
    For each site i, compute the inradius of its Voronoi cell clipped to [0,W]x[0,H].
    The inradius is the minimum over:
      - distance to left boundary: x_i
      - distance to right boundary: W - x_i
      - distance to bottom boundary: y_i
      - distance to top boundary: H - y_i
      - for each neighbor j: distance to the perpendicular bisector of i-j
        (only if the bisector actually forms a cell edge, but we use all as upper bounds)
    
    The perpendicular bisector between i and j: the set of points equidistant.
    Distance from site i to this bisector = |d_ij| / 2, where d_ij is the distance between i and j.
    BUT this is only the inradius contribution if the Voronoi edge actually exists.
    For a safe upper bound on the inradius, we take min over all j of d_ij/2 and boundary distances.
    This gives a valid (possibly non-tight) inradius that guarantees no overlap.
    """
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        r_i = min(x[i], W - x[i], y[i], H - y[i])
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.sqrt(dx*dx + dy*dy)
            r_i = min(r_i, d * 0.5)
        r[i] = max(r_i, 1e-12)
    return r

def _soft_min(values, temperature):
    """
    Smooth approximation of min via log-sum-exp.
    soft_min(v) = -log(sum(exp(-v_i / T))) * T
    """
    v = np.asarray(values, dtype=float)
    v = np.clip(v, 1e-12, 1e6)
    scaled = -v / temperature
    # Numerical stability
    max_s = np.max(scaled)
    if not np.isfinite(max_s):
        return np.min(v)
    exp_s = np.exp(scaled - max_s)
    s = np.sum(exp_s)
    if s <= 0 or not np.isfinite(s):
        return np.min(v)
    result = -temperature * (np.log(s) + max_s)
    if not np.isfinite(result) or result < 0:
        return np.min(v)
    return result

def _objective_smooth(vars, W, temperature):
    """
    Smooth objective: minimize -sum(soft_min of inradius components)
    vars: [x1, y1, ..., xN, yN]
    """
    H = 2.0 - W
    x = vars[0:N]
    y = vars[N:2*N]
    
    # Clip to valid range to keep things stable
    x = np.clip(x, 1e-4, W - 1e-4)
    y = np.clip(y, 1e-4, H - 1e-4)
    
    total = 0.0
    for i in range(N):
        # Boundary distances
        comps = [x[i], W - x[i], y[i], H - y[i]]
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.sqrt(dx*dx + dy*dy)
            comps.append(d * 0.5)
        r_i = _soft_min(comps, temperature)
        total += r_i
    
    return -total

def _objective_exact(vars, W):
    """
    Exact objective: minimize -sum(inradii)
    """
    H = 2.0 - W
    x = vars[0:N]
    y = vars[N:2*N]
    
    x = np.clip(x, 1e-4, W - 1e-4)
    y = np.clip(y, 1e-4, H - 1e-4)
    
    r = _voronoi_inradii(x, y, W, H)
    return -np.sum(r)

def _gradient_smooth(vars, W, temperature):
    """
    Numerical gradient (central differences) of the smooth objective.
    """
    H = 2.0 - W
    n = 2 * N
    grad = np.zeros(n)
    h = 1e-5
    for k in range(n):
        vp = vars.copy()
        vm = vars.copy()
        vp[k] += h
        vm[k] -= h
        # Clip to bounds
        for i in range(N):
            vp[i] = np.clip(vp[i], 1e-4, W - 1e-4)
            vp[N+i] = np.clip(vp[N+i], 1e-4, H - 1e-4)
            vm[i] = np.clip(vm[i], 1e-4, W - 1e-4)
            vm[N+i] = np.clip(vm[N+i], 1e-4, H - 1e-4)
        grad[k] = (_objective_smooth(vp, W, temperature) - _objective_smooth(vm, W, temperature)) / (2*h)
    return grad

def _compute_value_from_centers(x, y, W):
    """
    Given centers, compute the valid sum of Voronoi inradii.
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 0.0, None, None
    r = _voronoi_inradii(x, y, W, H)
    # Verify: check all circles are within bounds and non-overlapping
    for i in range(N):
        if x[i] < r[i] - 1e-9 or x[i] > W - r[i] + 1e-9:
            return 0.0, None, None
        if y[i] < r[i] - 1e-9 or y[i] > H - r[i] + 1e-9:
            return 0.0, None, None
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            min_d = r[i] + r[j]
            if d_sq < min_d*min_d - 1e-9:
                return 0.0, None, None
    return np.sum(r), x.copy(), y.copy(), r.copy()

def _grid_init_centers(W, H, n_cols, n_rows, noise_scale=0.0):
    """
    Initialize N centers in a grid pattern.
    """
    x = np.zeros(N)
    y = np.zeros(N)
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            x[idx] = (col + 0.5) * (W / n_cols)
            y[idx] = (row + 0.5) * (H / n_rows)
            idx += 1
        if idx >= N:
            break
    if noise_scale > 0:
        x += np.random.uniform(-noise_scale, noise_scale, N)
        y += np.random.uniform(-noise_scale, noise_scale, N)
    x = np.clip(x, 1e-3, W - 1e-3)
    y = np.clip(y, 1e-3, H - 1e-3)
    return x, y

def _run_center_optimization(x0, y0, W, deadline, temperature=0.01):
    """
    Optimize center positions to maximize sum of Voronoi inradii.
    Uses L-BFGS-B with smooth objective.
    """
    H = 2.0 - W
    vars0 = np.concatenate([x0, y0])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-4, W - 1e-4))
    for i in range(N):
        bounds.append((1e-4, H - 1e-4))
    
    options = {
        'maxiter': 300,
        'ftol': 1e-14,
        'gtol': 1e-10,
        'disp': False,
        'maxfun': 5000
    }
    
    start_time = time.time()
    
    try:
        res = minimize(
            lambda v: _objective_smooth(v, W, temperature),
            vars0,
            jac=lambda v: _gradient_smooth(v, W, temperature),
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        vars_opt = res.x
        x_opt = vars_opt[0:N]
        y_opt = vars_opt[N:2*N]
        
        val, xo, yo, ro = _compute_value_from_centers(x_opt, y_opt, W)
        if val > 0:
            return x_opt, y_opt, ro, W, val
        else:
            val0, _, _, _ = _compute_value_from_centers(x0, y0, W)
            return x0, y0, _voronoi_inradii(x0, y0, W, H), W, val0
            
    except Exception as e:
        val0, _, _, _ = _compute_value_from_centers(x0, y0, W)
        return x0, y0, _voronoi_inradii(x0, y0, W, H), W, val0

def _polish_centers(x, y, W, deadline):
    """
    Final high-precision polish with exact objective and tight tolerances.
    """
    H = 2.0 - W
    vars0 = np.concatenate([x, y])
    
    bounds = []
    for i in range(N):
        bounds.append((max(1e-4, x[i] - 0.005), min(W - 1e-4, x[i] + 0.005)))
    for i in range(N):
        bounds.append((max(1e-4, y[i] - 0.005), min(H - 1e-4, y[i] + 0.005)))
    
    options = {
        'maxiter': 500,
        'ftol': 1e-16,
        'gtol': 1e-12,
        'disp': False,
        'maxfun': 10000
    }
    
    start_time = time.time()
    
    try:
        res = minimize(
            lambda v: _objective_exact(v, W),
            vars0,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        vars_opt = res.x
        x_opt = vars_opt[0:N]
        y_opt = vars_opt[N:2*N]
        
        val, _, _, ro = _compute_value_from_centers(x_opt, y_opt, W)
        if val > 0:
            return x_opt, y_opt, ro, val
        else:
            val0, _, _, ro0 = _compute_value_from_centers(x, y, W)
            return x, y, ro0, val0
            
    except Exception as e:
        val0, _, _, ro0 = _compute_value_from_centers(x, y, W)
        return x, y, ro0, val0

def _repulsion_refine(x, y, W, deadline, steps=2000):
    """
    Simple gradient-free repulsion-based refinement.
    Each step: move each center slightly away from neighbors and toward center of its cell.
    """
    H = 2.0 - W
    x = x.copy()
    y = y.copy()
    
    best_val, _, _, _ = _compute_value_from_centers(x, y, W)
    best_x, best_y = x.copy(), y.copy()
    
    step_size = 0.005
    start_time = time.time()
    
    for step in range(steps):
        if time.time() > deadline - 10:
            break
        if step % 200 == 0:
            step_size *= 0.95  # Annealing
        
        dx = np.zeros(N)
        dy = np.zeros(N)
        
        for i in range(N):
            # Repulsion from neighbors
            for j in range(N):
                if i == j:
                    continue
                ddx = x[i] - x[j]
                ddy = y[i] - y[j]
                d_sq = ddx*ddx + ddy*ddy
                d = math.sqrt(d_sq) if d_sq > 1e-12 else 1e-6
                # Strong repulsion when close
                force = 1.0 / (d * d + 0.01)
                dx[i] += force * ddx / d
                dy[i] += force * ddy / d
            
            # Attraction to center of rectangle (weak)
            dx[i] += 0.01 * (W/2 - x[i])
            dy[i] += 0.01 * (H/2 - y[i])
        
        # Update
        for i in range(N):
            x[i] += step_size * dx[i]
            y[i] += step_size * dy[i]
            x[i] = np.clip(x[i], 1e-3, W - 1e-3)
            y[i] = np.clip(y[i], 1e-3, H - 1e-3)
        
        val, _, _, _ = _compute_value_from_centers(x, y, W)
        if val > best_val:
            best_val = val
            best_x = x.copy()
            best_y = y.copy()
    
    return best_x, best_y, _voronoi_inradii(best_x, best_y, W, H), best_val

def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s limit
    
    best_val = 0.0
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_r = np.zeros(N)
    best_W = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    # Grid configurations (cols, rows)
    grid_configs = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (3, 8), (8, 3), (4, 5), (5, 4), (6, 5)]
    
    # First pass: explore aspect ratios and grid configs
    for W0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        H0 = 2.0 - W0
        
        for n_cols, n_rows in grid_configs:
            if time.time() > deadline - 30:
                break
            
            x0, y0 = _grid_init_centers(W0, H0, n_cols, n_rows, noise_scale=0.0)
            
            x_opt, y_opt, r_opt, W_opt, val = _run_center_optimization(x0, y0, W0, deadline, temperature=0.01)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
    
    # Second pass: refine best with different temperatures and restarts
    if time.time() < deadline - 25 and best_val > 0:
        W_best = best_W
        H_best = 2.0 - W_best
        
        # Try different temperatures
        for temp in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > deadline - 25:
                break
            
            n_cols, n_rows = (5, 5) if W_best > 1.05 else (4, 6)
            x0, y0 = _grid_init_centers(W_best, H_best, n_cols, n_rows, noise_scale=0.01)
            
            x_opt, y_opt, r_opt, W_opt, val = _run_center_optimization(x0, y0, W_best, deadline, temperature=temp)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
        
        # Random restarts
        for _ in range(5):
            if time.time() > deadline - 25:
                break
            
            n_cols, n_rows = (5, 5) if W_best > 1.05 else (4, 6)
            x0, y0 = _grid_init_centers(W_best, H_best, n_cols, n_rows, noise_scale=0.03)
            
            x_opt, y_opt, r_opt, W_opt, val = _run_center_optimization(x0, y0, W_best, deadline, temperature=0.01)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
    
    # Repulsion-based refinement
    if time.time() < deadline - 15 and best_val > 0:
        x_rep, y_rep, r_rep, val_rep = _repulsion_refine(best_x, best_y, best_W, deadline, steps=1000)
        if val_rep > best_val:
            best_val = val_rep
            best_x = x_rep
            best_y = y_rep
            best_r = r_rep
    
    # Final polish pass
    if time.time() < deadline - 5 and best_val > 0:
        x_pol, y_pol, r_pol, val_pol = _polish_centers(best_x, best_y, best_W, deadline)
        if val_pol > best_val:
            best_x = x_pol
            best_y = y_pol
            best_r = r_pol
            best_val = val_pol
    
    # Fallback
    if best_val == 0.0:
        W = 1.0
        H = 1.0
        side = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            best_x[i] = (col + 0.5) / side
            best_y[i] = (row + 0.5) / side
            best_r[i] = 0.5 / side - 1e-4
        best_W = W
    
    return np.column_stack([best_x, best_y, best_r])

# EVOLVE-BLOCK-END
