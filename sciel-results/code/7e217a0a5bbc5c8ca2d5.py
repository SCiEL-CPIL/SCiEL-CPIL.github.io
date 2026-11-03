import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    return np.logaddexp(0, beta * x) / beta

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus."""
    return 1.0 / (1.0 + np.exp(-beta * x))

def _objective_and_grad(v, beta=100.0):
    """
    Objective: maximize sum of radii (all equal r) subject to constraints.
    Variables: v = [w, r, x_0..x_20, y_0..y_20]
    h = 2 - w (perimeter 4 means 2(w+h)=4, so w+h=2)
    """
    w = v[0]
    r = v[1]
    h = 2.0 - w
    xs = v[2:2+N]
    ys = v[2+N:2+2*N]
    
    # Penalty terms
    penalty = 0.0
    grad_penalty = np.zeros_like(v)
    
    # Boundary constraints: r_i >= r, x_i <= w-r, y_i >= r, y_i <= h-r
    # We want x_i - r >= 0, w - x_i - r >= 0, y_i - r >= 0, h - y_i - r >= 0
    for i in range(N):
        # Left: x_i - r >= 0
        c = xs[i] - r
        penalty += _softplus(-c, beta)
        g = -_softplus_grad(-c, beta)
        grad_penalty[2+i] += g
        grad_penalty[1] -= g
        
        # Right: w - x_i - r >= 0
        c = w - xs[i] - r
        penalty += _softplus(-c, beta)
        g = -_softplus_grad(-c, beta)
        grad_penalty[0] += g
        grad_penalty[2+i] -= g
        grad_penalty[1] -= g
        
        # Bottom: y_i - r >= 0
        c = ys[i] - r
        penalty += _softplus(-c, beta)
        g = -_softplus_grad(-c, beta)
        grad_penalty[2+N+i] += g
        grad_penalty[1] -= g
        
        # Top: h - y_i - r >= 0  =>  2 - w - y_i - r >= 0
        c = 2.0 - w - ys[i] - r
        penalty += _softplus(-c, beta)
        g = -_softplus_grad(-c, beta)
        grad_penalty[0] -= g
        grad_penalty[2+N+i] -= g
        grad_penalty[1] -= g
    
    # Overlap constraints: ||p_i - p_j|| >= 2r
    # dist^2 - (2r)^2 >= 0, but we use dist - 2r >= 0 with sqrt
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-30)
            c = dist - 2.0*r
            if c < 0.05:  # Only compute penalty when close
                penalty += _softplus(-c, beta)
                g = -_softplus_grad(-c, beta)
                # d(dist)/dx_i = dx/dist, d(dist)/dy_i = dy/dist
                grad_penalty[2+i] += g * dx / dist
                grad_penalty[2+N+i] += g * dy / dist
                grad_penalty[2+j] -= g * dx / dist
                grad_penalty[2+N+j] -= g * dy / dist
                grad_penalty[1] -= 2.0 * g
    
    # Objective: maximize r, so minimize -r
    obj = -r + penalty
    
    grad = np.zeros_like(v)
    grad[1] = -1.0  # d(-r)/dr
    grad += grad_penalty
    
    return obj, grad

def _strictly_valid(w, h, r, xs, ys, tol=1e-7):
    """Check if configuration is strictly valid."""
    if w <= 0 or h <= 0 or r <= 0:
        return False
    if w + h > 2.0 + tol:
        return False
    for i in range(N):
        if xs[i] < r - tol or xs[i] > w - r + tol:
            return False
        if ys[i] < r - tol or ys[i] > h - r + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < 2.0*r - tol:
                return False
    return True

def _hex_init(rows):
    """Generate hexagonal packing initial configuration."""
    pts = []
    for k, n_c in enumerate(rows):
        y_k = k * np.sqrt(3)
        offset = 1.0 if k % 2 == 1 else 0.0
        for i in range(n_c):
            x_k = i * 2.0 + offset
            pts.append((x_k, y_k))
    
    pts = np.array(pts[:N])
    if len(pts) < N:
        # Pad if needed
        while len(pts) < N:
            pts = np.vstack([pts, pts[-1:] + np.array([0.1, 0.1])])
    
    center = pts.mean(axis=0)
    pts -= center
    
    xs = pts[:, 0]
    ys = pts[:, 1]
    w_c = xs.max() - xs.min()
    h_c = ys.max() - ys.min()
    
    w_total = w_c + 2.0
    h_total = h_c + 2.0
    
    scale = 2.0 / (w_total + h_total)
    r = scale
    w = scale * w_total
    h = scale * h_total
    
    xs_final = pts[:, 0] * scale + w/2.0
    ys_final = pts[:, 1] * scale + h/2.0
    
    return w, r, xs_final, ys_final

def _grid_init(rows, cols):
    """Generate grid packing initial configuration."""
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
        if len(pts) >= N:
            break
    
    pts = np.array(pts[:N])
    center = pts.mean(axis=0)
    pts -= center
    
    xs = pts[:, 0]
    ys = pts[:, 1]
    w_c = xs.max() - xs.min()
    h_c = ys.max() - ys.min()
    
    w_total = w_c + 2.0
    h_total = h_c + 2.0
    
    scale = 2.0 / (w_total + h_total)
    r = scale
    w = scale * w_total
    h = scale * h_total
    
    xs_final = pts[:, 0] * scale + w/2.0
    ys_final = pts[:, 1] * scale + h/2.0
    
    return w, r, xs_final, ys_final

def _random_init(seed=None):
    """Generate random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    w = 1.0
    h = 1.0
    r = 0.03
    xs = rng.uniform(r, w-r, N)
    ys = rng.uniform(r, h-r, N)
    return w, r, xs, ys

def _optimize_from(w, r, xs, ys, beta_start=20.0, beta_end=200.0, n_steps=8, time_limit=None, deadline=None):
    """
    Run L-BFGS-B optimization with annealed beta.
    Returns best (w, r, xs, ys) found.
    """
    best_obj = np.inf
    best_v = None
    
    v = np.concatenate([[w, r], xs, ys])
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (step / max(n_steps - 1, 1))
        
        def obj_func(v):
            return _objective_and_grad(v, beta)[0]
        
        def grad_func(v):
            return _objective_and_grad(v, beta)[1]
        
        bounds = [(0.1, 1.9), (0.01, 0.5)] + [(-0.5, 2.5)] * (2*N)
        
        try:
            res = minimize(obj_func, v, jac=grad_func, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x.copy()
            
            v = res.x.copy()
        except Exception:
            break
    
    if best_v is not None:
        w_best = best_v[0]
        r_best = best_v[1]
        xs_best = best_v[2:2+N]
        ys_best = best_v[2+N:2+2*N]
        
        if _strictly_valid(w_best, 2.0 - w_best, r_best, xs_best, ys_best):
            return w_best, r_best, xs_best, ys_best
    
    return w, r, xs, ys

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    def check_time():
        return time.time() > deadline
    
    best_r = 0
    best_config = None
    
    # Initial configurations
    inits = []
    
    # Hexagonal packings
    inits.append(_hex_init([7, 7, 7]))
    inits.append(_hex_init([4, 5, 4, 4, 4]))
    inits.append(_hex_init([5, 4, 5, 4, 3]))
    inits.append(_hex_init([3, 5, 4, 5, 4]))
    inits.append(_hex_init([6, 6, 6, 3]))
    inits.append(_hex_init([4, 4, 4, 4, 5]))
    inits.append(_hex_init([5, 5, 5, 3, 3]))
    
    # Grid packings
    inits.append(_grid_init(5, 5))
    inits.append(_grid_init(3, 7))
    inits.append(_grid_init(4, 6))
    inits.append(_grid_init(7, 3))
    inits.append(_grid_init(6, 4))
    
    # Random inits
    for s in range(10):
        inits.append(_random_init(seed=s))
    
    # Process each initial configuration
    for idx, (w, r, xs, ys) in enumerate(inits):
        if check_time():
            break
        
        # Optimize with annealed beta
        w_opt, r_opt, xs_opt, ys_opt = _optimize_from(w, r, xs, ys, 
                                                       beta_start=20.0, beta_end=200.0,
                                                       n_steps=6, deadline=deadline)
        
        if r_opt > best_r and _strictly_valid(w_opt, 2.0 - w_opt, r_opt, xs_opt, ys_opt):
            best_r = r_opt
            best_config = (w_opt, r_opt, xs_opt, ys_opt)
        
        # If we have time, do a high-precision polish on the best
        if check_time():
            break
    
    # Final high-precision polish on the best configuration
    if best_config is not None and not check_time():
        w, r, xs, ys = best_config
        w_pol, r_pol, xs_pol, ys_pol = _optimize_from(w, r, xs, ys,
                                                       beta_start=100.0, beta_end=500.0,
                                                       n_steps=4, deadline=deadline)
        if r_pol > best_r and _strictly_valid(w_pol, 2.0 - w_pol, r_pol, xs_pol, ys_pol):
            best_r = r_pol
            best_config = (w_pol, r_pol, xs_pol, ys_pol)
    
    # If no valid config found, use fallback
    if best_config is None:
        w, r, xs, ys = inits[0]
        # Scale down slightly to ensure validity
        r *= 0.99
        xs = xs * 0.99 + w * 0.005
        ys = ys * 0.99 + (2.0 - w) * 0.005
        best_config = (w, r, xs, ys)
        best_r = r
    
    w, r, xs, ys = best_config
    
    # Final safety: shrink radius slightly
    r_final = r * (1 - 1e-5)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
