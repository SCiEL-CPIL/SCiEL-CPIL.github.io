import numpy as np
from scipy.optimize import minimize
import time

N = 21
_DEADLINE = 130.0  # safety margin before 150s kill

def _compute_radii_from_voronoi(x, y, W, H):
    """Compute the maximum radius for each circle given center positions,
    using the Voronoi cell inradii approximation (distance to nearest neighbor/2
    and distance to boundary)."""
    r = np.zeros(N)
    for i in range(N):
        # Distance to boundary
        d_boundary = min(x[i], W - x[i], y[i], H - y[i])
        # Distance to nearest other circle center
        min_dist = np.inf
        for j in range(N):
            if i == j:
                continue
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < min_dist:
                min_dist = d
        # The radius is limited by half the distance to nearest neighbor and boundary
        r[i] = min(d_boundary, 0.5 * min_dist)
    return r

def _compute_radii_vec(x, y, W, H):
    """Vectorized computation of max radii for each circle."""
    # Distance to boundary for each circle
    d_boundary = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist, axis=1)
    r = np.minimum(d_boundary, 0.5 * min_dist)
    return r

def _sum_radii(x, y, W, H):
    """Quick sum of radii using vectorized computation."""
    return np.sum(_compute_radii_vec(x, y, W, H))

def _smooth_overlap_penalty(x, y, r, W, H, alpha=50.0):
    """Smooth penalty for overlaps and boundary violations using log-sum-exp."""
    # Boundary penalties (smooth)
    pen = 0.0
    # x boundaries
    for i in range(N):
        pen += np.logaddexp(0, -alpha * (x[i] - r[i]))  # x < r
        pen += np.logaddexp(0, -alpha * ((W - r[i]) - x[i]))  # x > W - r
    # y boundaries
    for i in range(N):
        pen += np.logaddexp(0, -alpha * (y[i] - r[i]))
        pen += np.logaddexp(0, -alpha * ((H - r[i]) - y[i]))
    
    # Pairwise overlap penalties (smooth)
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-12
            req = r[i] + r[j]
            # Smooth penalty when dist < req
            pen += np.logaddexp(0, -alpha * (dist - req))
    
    return pen

def _sa_optimize(x, y, W, H, start_time, time_budget, rng):
    """Simulated Annealing to find good circle positions.
    Returns best (x, y, r, val) found."""
    r = _compute_radii_vec(x, y, W, H)
    best_val = np.sum(r)
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    
    # SA parameters
    T0 = 0.05
    T_end = 0.001
    n_steps = 50000
    # Adaptive step size
    step_scale = 0.1
    
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining <= 0:
        return best_x, best_y, best_r, best_val
    
    # Estimate time per step
    steps_done = 0
    T = T0
    cooling = (T_end / T0) ** (1.0 / max(n_steps, 1))
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Propose a random move: pick a random circle and displace it
        i = rng.integers(0, N)
        scale = step_scale * (0.5 + 0.5 * (1.0 - step / n_steps))
        dx = rng.uniform(-scale, scale) * W
        dy = rng.uniform(-scale, scale) * H
        
        x_new = x.copy()
        x_new[i] = np.clip(x[i] + dx, 0.01*W, 0.99*W)
        y_new = y.copy()
        y_new[i] = np.clip(y[i] + dy, 0.01*H, 0.99*H)
        
        r_new = _compute_radii_vec(x_new, y_new, W, H)
        val_new = np.sum(r_new)
        val_old = np.sum(_compute_radii_vec(x, y, W, H))
        
        delta = val_new - val_old
        if delta > 0 or rng.random() < np.exp(delta / T):
            x, y, r = x_new, y_new, r_new
            if val_new > best_val:
                best_val = val_new
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        
        T *= cooling
        steps_done += 1
    
    return best_x, best_y, best_r, best_val

def _validate(x, y, r, W, H, tol=1e-5):
    """Check if the packing is valid."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _optimize_for_aspect(W, H, start_time, time_budget, rng):
    """Optimize circle positions for a given aspect ratio using SA + L-BFGS-B polish."""
    best_val = -1
    best_x = None
    best_y = None
    best_r = None
    
    # Precompute pair indices for efficiency
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Simulated Annealing exploration
    # Use multiple SA runs with different initializations
    n_sa_inits = 3
    sa_time_fraction = 0.6  # Use 60% of time for SA
    
    for init_idx in range(n_sa_inits):
        if time.time() - start_time > time_budget * sa_time_fraction:
            break
        
        if init_idx == 0:
            # Grid initialization
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
        else:
            # Random initialization with jitter
            cols = int(np.sqrt(N))
            rows = int(np.ceil(N / cols))
            x = np.zeros(N)
            y = np.zeros(N)
            k = 0
            for i in range(cols):
                for j in range(rows):
                    if k >= N:
                        break
                    x[k] = (i + 0.5) * W / cols + rng.uniform(-0.15, 0.15) * W / cols
                    y[k] = (j + 0.5) * H / rows + rng.uniform(-0.15, 0.15) * H / rows
                    k += 1
            # Clip to valid range
            x = np.clip(x, 0.05*W, 0.95*W)
            y = np.clip(y, 0.05*H, 0.95*H)
        
        # Run SA
        sa_budget = time_budget * sa_time_fraction
        x_sa, y_sa, r_sa, val_sa = _sa_optimize(x, y, W, H, start_time, sa_budget, rng)
        
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()
    
    # Phase 2: L-BFGS-B polish on the best SA result
    if best_x is not None and time.time() - start_time < time_budget * 0.95:
        # Objective for L-BFGS-B polish
        def objective(v):
            x_c = v[0:N]
            y_c = v[N:2*N]
            
            # Compute radii from Voronoi approximation
            r_c = _compute_radii_from_voronoi(x_c, y_c, W, H)
            
            # Maximize sum of radii (minimize negative)
            val = -np.sum(r_c)
            
            # Smooth penalty for overlaps
            pen = _smooth_overlap_penalty(x_c, y_c, r_c, W, H, alpha=50.0)
            
            return val + 5.0 * pen
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01*W, 0.99*W))
            bounds.append((0.01*H, 0.99*H))
        
        x0 = np.concatenate([best_x, best_y])
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception:
            pass
        else:
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            r_opt = _compute_radii_from_voronoi(x_opt, y_opt, W, H)
            
            # Validate
            if _validate(x_opt, y_opt, r_opt, W, H):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_x = x_opt.copy()
                    best_y = y_opt.copy()
                    best_r = r_opt.copy()
    
    if best_x is None:
        return None, None, None, -1
    return best_x, best_y, best_r, best_val

def _local_refine(x, y, r, W, H, start_time, time_budget):
    """Final high-precision polish pass."""
    if time.time() - start_time > time_budget:
        return x, y, r
    
    # Try to refine with tighter tolerances
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def objective(v):
        x_c = v[0:N]
        y_c = v[N:2*N]
        r_c = _compute_radii_from_voronoi(x_c, y_c, W, H)
        
        val = -np.sum(r_c)
        
        # Harder penalty for violations
        pen = 0.0
        for i in range(N):
            pen += max(0, r_c[i] - x_c[i])**2
            pen += max(0, x_c[i] - (W - r_c[i]))**2
            pen += max(0, r_c[i] - y_c[i])**2
            pen += max(0, y_c[i] - (H - r_c[i]))**2
        for i, j in pairs:
            dist = np.sqrt((x_c[i]-x_c[j])**2 + (y_c[i]-y_c[j])**2)
            req = r_c[i] + r_c[j]
            if dist < req:
                pen += (req - dist)**2
        
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
        y_ref = res.x[N:2*N]
        r_ref = _compute_radii_from_voronoi(x_ref, y_ref, W, H)
        
        if _validate(x_ref, y_ref, r_ref, W, H) and np.sum(r_ref) >= np.sum(r) - 1e-8:
            return x_ref, y_ref, r_ref
    except Exception:
        pass
    
    return x, y, r

def construct():
    start_time = time.time()
    time_budget = _DEADLINE - start_time - 5.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_val = -1
    best_res = None
    
    # Try different aspect ratios - focus on more promising ones
    # Perimeter 4 means W + H = 2
    # Square is W=H=1. Try a range but weight toward square
    widths = [0.8, 0.9, 1.0, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]
    
    # Allocate time per aspect ratio
    n_widths = len(widths)
    per_aspect_budget = time_budget / n_widths
    
    for idx, W in enumerate(widths):
        H = 2.0 - W
        if H <= 0:
            continue
        
        if time.time() - start_time > time_budget:
            break
        
        # Give later aspects less time if we're running low
        remaining = time_budget - (time.time() - start_time)
        aspect_budget = min(per_aspect_budget, remaining * 0.9)
        
        if aspect_budget < 2.0:  # Not enough time for meaningful optimization
            break
        
        x, y, r, val = _optimize_for_aspect(W, H, start_time, aspect_budget, rng)
        
        if x is not None and val > best_val:
            # Try local refinement with remaining time
            refine_budget = min(time_budget - (time.time() - start_time) - 1.0, 10.0)
            if refine_budget > 1.0:
                x, y, r = _local_refine(x, y, r, W, H, start_time, time_budget)
                val = np.sum(r)
            if val > best_val:
                best_val = val
                best_res = np.column_stack((x, y, r))
    
    if best_res is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_res
