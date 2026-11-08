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

def _optimize_for_aspect(W, H, start_time, time_budget):
    """Optimize circle positions for a given aspect ratio using Voronoi-based radii."""
    best_val = -1
    best_x = None
    best_y = None
    best_r = None
    
    # Precompute pair indices for efficiency
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Multiple initializations
    n_inits = 3
    for init_idx in range(n_inits):
        if time.time() - start_time > time_budget:
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
                    x[k] = (i + 0.5) * W / cols + np.random.uniform(-0.1, 0.1) * W / cols
                    y[k] = (j + 0.5) * H / rows + np.random.uniform(-0.1, 0.1) * H / rows
                    k += 1
            # Clip to valid range
            x = np.clip(x, 0.05*W, 0.95*W)
            y = np.clip(y, 0.05*H, 0.95*H)
        
        # Objective: maximize sum of Voronoi-derived radii
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
        
        # Bounds: centers must be strictly inside
        bounds = [(0.01*W, 0.99*W), (0.01*H, 0.99*H)] * N
        # Actually we need 2N bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01*W, 0.99*W))
            bounds.append((0.01*H, 0.99*H))
        
        x0 = np.concatenate([x, y])
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception:
            continue
        
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
            # Try local refinement
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
