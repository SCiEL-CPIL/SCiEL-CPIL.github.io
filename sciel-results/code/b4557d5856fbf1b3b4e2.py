import numpy as np
from scipy.optimize import minimize
import time

N = 21
_DEADLINE = 130.0  # safety margin before 150s kill

def _compute_radii_vec(x, y, W, H):
    """Vectorized computation of max radii from center positions."""
    # Distance to boundary
    d_boundary = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist, axis=1)
    r = np.minimum(d_boundary, 0.5 * min_dist)
    return r

def _compute_radii_from_voronoi(x, y, W, H):
    """Compute the maximum radius for each circle given center positions."""
    return _compute_radii_vec(x, y, W, H)

def _smooth_overlap_penalty(x, y, r, W, H, alpha=50.0):
    """Smooth penalty for overlaps and boundary violations using log-sum-exp."""
    pen = 0.0
    for i in range(N):
        pen += np.logaddexp(0, -alpha * (x[i] - r[i]))
        pen += np.logaddexp(0, -alpha * ((W - r[i]) - x[i]))
        pen += np.logaddexp(0, -alpha * (y[i] - r[i]))
        pen += np.logaddexp(0, -alpha * ((H - r[i]) - y[i]))
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            req = r[i] + r[j]
            pen += np.logaddexp(0, -alpha * (dist - req))
    return pen

def _validate(x, y, r, W, H, tol=1e-5):
    """Check if the packing is valid."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _initial_grid(W, H):
    """Generate grid initialization."""
    cols = int(np.sqrt(N))
    rows = int(np.ceil(N / cols))
    x = np.zeros(N)
    y = np.zeros(N)
    k = 0
    for i in range(cols):
        for j in range(rows):
            if k >= N:
                break
            x[k] = (i + 0.5) * W / cols
            y[k] = (j + 0.5) * H / rows
            k += 1
    return x, y

def _initial_random(W, H, seed=42):
    """Generate random initialization with jitter from grid."""
    rng = np.random.RandomState(seed)
    x, y = _initial_grid(W, H)
    cols = int(np.sqrt(N))
    rows = int(np.ceil(N / cols))
    for k in range(N):
        i = k % cols
        j = k // cols
        x[k] = (i + 0.5) * W / cols + rng.uniform(-0.15, 0.15) * W / cols
        y[k] = (j + 0.5) * H / rows + rng.uniform(-0.15, 0.15) * H / rows
    x = np.clip(x, 0.05 * W, 0.95 * W)
    y = np.clip(y, 0.05 * H, 0.95 * H)
    return x, y

def _sa_optimize(x, y, W, H, start_time, time_budget, n_inits=3):
    """Simulated Annealing optimization for center positions."""
    best_val = -1
    best_x, best_y, best_r = None, None, None
    
    for init_idx in range(n_inits):
        if time.time() - start_time > time_budget:
            break
        
        if init_idx == 0:
            x_cur, y_cur = _initial_grid(W, H)
        else:
            x_cur, y_cur = _initial_random(W, H, seed=100 + init_idx * 7)
        
        r_cur = _compute_radii_vec(x_cur, y_cur, W, H)
        cur_val = np.sum(r_cur)
        
        best_x, best_y, best_r = x_cur.copy(), y_cur.copy(), r_cur.copy()
        best_val = cur_val
        
        # SA parameters
        T_start = 0.01 * max(W, H)
        T_end = 1e-6
        n_iter = 5000
        cooling = (T_end / T_start) ** (1.0 / n_iter)
        
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        for it in range(n_iter):
            if time.time() - start_time > time_budget:
                break
            
            T = T_start * (cooling ** it)
            
            # Pick a random circle and perturb it
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.02 * max(W, H))
            dy = np.random.normal(0, 0.02 * max(W, H))
            
            x_new = x_cur.copy()
            y_new = y_cur.copy()
            x_new[idx] += dx
            y_new[idx] += dy
            
            # Keep within bounds
            x_new[idx] = np.clip(x_new[idx], 0.01 * W, 0.99 * W)
            y_new[idx] = np.clip(y_new[idx], 0.01 * H, 0.99 * H)
            
            r_new = _compute_radii_vec(x_new, y_new, W, H)
            new_val = np.sum(r_new)
            
            delta = new_val - cur_val
            
            if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
                x_cur, y_cur = x_new, y_new
                r_cur = r_new
                cur_val = new_val
                
                if cur_val > best_val:
                    best_val = cur_val
                    best_x, best_y, best_r = x_cur.copy(), y_cur.copy(), r_cur.copy()
        
        # Gradient polish after SA
        if time.time() - start_time < time_budget:
            x_pol, y_pol, r_pol, val_pol = _gradient_polish(
                best_x, best_y, W, H, start_time, time_budget)
            if val_pol > best_val:
                best_val = val_pol
                best_x, best_y, best_r = x_pol, y_pol, r_pol
    
    if best_x is None:
        return None, None, None, -1
    return best_x, best_y, best_r, best_val

def _gradient_polish(x, y, W, H, start_time, time_budget):
    """Gradient-based polish using L-BFGS-B with smooth penalty."""
    def objective(v):
        x_c = v[0:N]
        y_c = v[N:2 * N]
        r_c = _compute_radii_vec(x_c, y_c, W, H)
        val = -np.sum(r_c)
        pen = _smooth_overlap_penalty(x_c, y_c, r_c, W, H, alpha=50.0)
        return val + 5.0 * pen
    
    bounds = []
    for i in range(N):
        bounds.append((0.01 * W, 0.99 * W))
        bounds.append((0.01 * H, 0.99 * H))
    
    x0 = np.concatenate([x, y])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        x_opt = res.x[0:N]
        y_opt = res.x[N:2 * N]
        r_opt = _compute_radii_vec(x_opt, y_opt, W, H)
        
        if _validate(x_opt, y_opt, r_opt, W, H):
            return x_opt, y_opt, r_opt, np.sum(r_opt)
    except Exception:
        pass
    
    return x, y, _compute_radii_vec(x, y, W, H), np.sum(_compute_radii_vec(x, y, W, H))

def _local_refine(x, y, r, W, H, start_time, time_budget):
    """Final high-precision polish pass with tighter tolerances."""
    if time.time() - start_time > time_budget:
        return x, y, r
    
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def objective(v):
        x_c = v[0:N]
        y_c = v[N:2 * N]
        r_c = _compute_radii_vec(x_c, y_c, W, H)
        val = -np.sum(r_c)
        pen = 0.0
        for i in range(N):
            pen += max(0, r_c[i] - x_c[i]) ** 2
            pen += max(0, x_c[i] - (W - r_c[i])) ** 2
            pen += max(0, r_c[i] - y_c[i]) ** 2
            pen += max(0, y_c[i] - (H - r_c[i])) ** 2
        for i, j in pairs:
            dist = np.sqrt((x_c[i] - x_c[j]) ** 2 + (y_c[i] - y_c[j]) ** 2)
            req = r_c[i] + r_c[j]
            if dist < req:
                pen += (req - dist) ** 2
        return val + 100.0 * pen
    
    x0 = np.concatenate([x, y])
    bounds = []
    for i in range(N):
        bounds.append((1e-4, W - 1e-4))
        bounds.append((1e-4, H - 1e-4))
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
        x_ref = res.x[0:N]
        y_ref = res.x[N:2 * N]
        r_ref = _compute_radii_vec(x_ref, y_ref, W, H)
        
        if _validate(x_ref, y_ref, r_ref, W, H) and np.sum(r_ref) >= np.sum(r) - 1e-8:
            return x_ref, y_ref, r_ref
    except Exception:
        pass
    
    return x, y, r

def _optimize_for_aspect(W, H, start_time, time_budget):
    """Optimize circle positions for a given aspect ratio."""
    # First try gradient-based optimization
    x_grid, y_grid = _initial_grid(W, H)
    x, y, r, val = _gradient_polish(x_grid, y_grid, W, H, start_time, time_budget)
    
    if x is None or val < 0:
        return None, None, None, -1
    
    best_val = val
    best_x, best_y, best_r = x, y, r
    
    # Then try SA for potentially better global solution
    if time.time() - start_time < time_budget:
        sa_x, sa_y, sa_r, sa_val = _sa_optimize(x, y, W, H, start_time, time_budget, n_inits=2)
        if sa_x is not None and sa_val > best_val:
            best_val = sa_val
            best_x, best_y, best_r = sa_x, sa_y, sa_r
    
    # Final refinement
    if time.time() - start_time < time_budget:
        best_x, best_y, best_r = _local_refine(best_x, best_y, best_r, W, H, start_time, time_budget)
        best_val = np.sum(best_r)
    
    if best_x is None:
        return None, None, None, -1
    return best_x, best_y, best_r, best_val

def construct():
    start_time = time.time()
    time_budget = _DEADLINE - start_time - 5.0  # 5s safety margin
    
    best_val = -1
    best_res = None
    
    # Try different aspect ratios
    widths = np.linspace(0.6, 1.4, 15)
    
    for W in widths:
        H = 2.0 - W
        if H <= 0:
            continue
        
        if time.time() - start_time > time_budget:
            break
        
        x, y, r, val = _optimize_for_aspect(W, H, start_time, time_budget)
        
        if x is not None and val > best_val:
            best_val = val
            best_res = np.column_stack((x, y, r))
    
    if best_res is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_res
