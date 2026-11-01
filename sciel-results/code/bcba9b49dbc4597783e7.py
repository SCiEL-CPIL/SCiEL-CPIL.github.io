import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0  # Safety margin before 150s limit

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if configuration is strictly valid."""
    if np.any(r <= 0):
        return False
    if w <= 0 or h <= 0:
        return False
    if 2*(w+h) > 4.0 + tol:
        return False
        
    # Boundary check
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
        
    # Overlap check
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    
    # Check all pairs i < j
    diff = dist_sq - r_sum**2
    np.fill_diagonal(diff, 1.0)
    
    if np.min(diff) < -tol:
        return False
        
    return True

def _build_objective(w, h):
    """
    Build objective and constraint functions for SLSQP optimization.
    Variables: [x_0..x_20, y_0..y_20, r_0..r_20] (63 variables)
    """
    def objective(v):
        x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
        # Maximize sum of radii => minimize negative sum
        return -np.sum(r)
    
    def constraints(v):
        x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
        cons = []
        
        # Boundary constraints: r_i <= x_i <= w - r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (w - v[i]) - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (h - v[N+i]) - v[2*N+i]})
        
        # Overlap constraints: dist_ij >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: 
                        (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
        
        return cons
    
    return objective, constraints

def _initialize_grid(w, h, seed=0):
    """
    Initialize circles in a grid pattern with some randomness.
    Returns initial variable vector [x, y, r].
    """
    rng = np.random.default_rng(seed)
    
    # Try different grid configurations
    configs = [
        (5, 5),  # 5x5 grid
        (4, 6),  # 4x6 grid  
        (6, 4),  # 6x4 grid
        (3, 7),  # 3x7 grid
        (7, 3),  # 7x3 grid
        (4, 5),  # 4x5 grid (20 spots, add 1 more)
    ]
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Pick a random config
    rows, cols = configs[rng.integers(len(configs))]
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            # Add small random perturbation
            cx += rng.uniform(-0.02*w, 0.02*w)
            cy += rng.uniform(-0.02*h, 0.02*h)
            x[idx] = np.clip(cx, 0.01, w-0.01)
            y[idx] = np.clip(cy, 0.01, h-0.01)
            # Initial radius: small fraction of cell size
            cell_w = w / cols
            cell_h = h / rows
            r[idx] = 0.3 * min(cell_w, cell_h)
            idx += 1
    
    # If we didn't fill all N, add remaining at random positions
    while idx < N:
        x[idx] = rng.uniform(0.05*w, 0.95*w)
        y[idx] = rng.uniform(0.05*h, 0.95*h)
        r[idx] = 0.05 * min(w, h)
        idx += 1
    
    # Ensure initial feasibility by shrinking radii if needed
    scale = 1.0
    for _ in range(50):
        if _is_valid(x, y, r * scale, w, h, tol=1e-6):
            break
        scale *= 0.95
    r *= scale
    
    return np.concatenate([x, y, r])

def _optimize_single(w, h, v0, max_iter=500, timeout=None):
    """
    Run SLSQP optimization from initial point v0.
    Returns optimized variable vector or None if failed.
    """
    if timeout is not None and time.time() > timeout:
        return None
    
    objective, constraints = _build_objective(w, h)
    
    # Bounds for each variable
    bounds = []
    for i in range(N):
        bounds.append((0.01, w - 0.01))  # x_i
    for i in range(N):
        bounds.append((0.01, h - 0.01))  # y_i
    for i in range(N):
        bounds.append((0.01, min(w, h)/2))  # r_i
    
    try:
        result = minimize(
            objective,
            v0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        
        if result.success and time.time() < (timeout or DEADLINE):
            return result.x
        else:
            # Even if not fully converged, check if result is valid
            x, y, r = result.x[:N], result.x[N:2*N], result.x[2*N:3*N]
            if _is_valid(x, y, r, w, h, tol=1e-8):
                return result.x
            return None
            
    except Exception:
        return None

def _polish_with_local_search(v, w, h, num_iters=100):
    """
    Local search polish: try small perturbations to improve the solution.
    """
    best_v = v.copy()
    best_val = -best_v[2*N:3*N].sum()
    
    for it in range(num_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # Try random small perturbations
        for _ in range(10):
            v_trial = best_v.copy()
            # Perturb positions slightly
            v_trial[:2*N] += np.random.uniform(-0.01, 0.01, 2*N)
            # Perturb radii slightly
            v_trial[2*N:] *= (1 + np.random.uniform(-0.02, 0.02, N))
            
            x, y, r = v_trial[:N], v_trial[N:2*N], v_trial[2*N:3*N]
            if _is_valid(x, y, r, w, h, tol=1e-9):
                val = -r.sum()
                if val < best_val:  # We're minimizing negative sum
                    best_val = val
                    best_v = v_trial
    
    return best_v

def construct():
    start_time = time.time()
    best_value = 0.0
    best_circles = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1, 0.6, 1.4, 0.85, 1.15, 0.75, 1.25]
    
    rng = np.random.default_rng(42)
    rng.shuffle(aspect_ratios)
    
    # Also try some random aspect ratios
    for _ in range(8):
        aspect_ratios.append(rng.uniform(0.5, 1.5))
        
    for w in aspect_ratios:
        if time.time() > DEADLINE - 20:
            break
            
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue
            
        # Multiple random restarts for this aspect ratio
        for restart in range(3):
            if time.time() > DEADLINE - 15:
                break
                
            # Initialize
            v0 = _initialize_grid(w, h, seed=restart * 100 + int(w * 100))
            
            # Optimize
            v_opt = _optimize_single(w, h, v0, max_iter=300, timeout=DEADLINE - 15)
            
            if v_opt is not None:
                x, y, r = v_opt[:N], v_opt[N:2*N], v_opt[2*N:3*N]
                val = r.sum()
                
                # Polish with local search
                if time.time() < DEADLINE - 10:
                    v_opt = _polish_with_local_search(v_opt, w, h, num_iters=50)
                    x, y, r = v_opt[:N], v_opt[N:2*N], v_opt[2*N:3*N]
                    val = r.sum()
                
                if val > best_value:
                    best_value = val
                    best_circles = np.column_stack((x, y, r))
                    
    if best_circles is None:
        # Fallback: Simple grid in square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
