"""Circle packing in perimeter-4 rectangle: force-directed + gradient refinement."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _clamp_to_rect(x, y, r, w, h, eps=1e-10):
    """Ensure circles stay inside rectangle."""
    x = np.maximum(r + eps, np.minimum(w - r - eps, x))
    y = np.maximum(r + eps, np.minimum(h - r - eps, y))
    return x, y

def _check_valid(x, y, r, w, h, tol=1e-8):
    """Check if packing is valid."""
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _force_directed_relax(x, y, r, w, h, max_iter=2000, dt=0.005, deadline=None):
    """
    Force-directed relaxation: push overlapping circles apart, push toward walls.
    Then grow all radii uniformly and re-relax.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    force = (min_d - d) * 2.0
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d
                elif d <= 1e-12:
                    # Random nudge for coincident points
                    fx[i] += random.uniform(-0.1, 0.1)
                    fy[i] += random.uniform(-0.1, 0.1)
                    fx[j] -= random.uniform(-0.1, 0.1)
                    fy[j] -= random.uniform(-0.1, 0.1)
        
        # Wall forces (keep circles inside)
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 5.0
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 5.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 5.0
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 5.0
        
        # Apply forces
        x += dt * fx
        y += dt * fy
        
        # Clamp
        x, y = _clamp_to_rect(x, y, r, w, h)
    
    # After relaxation, try to grow radii
    for grow_iter in range(100):
        if deadline is not None and time.time() > deadline:
            break
        # Try increasing all radii by a small amount
        dr = 0.001
        r_new = r + dr
        # Check if still valid after clamping
        x_new, y_new = _clamp_to_rect(x, y, r_new, w, h)
        
        # Quick overlap check
        valid = True
        for i in range(N):
            if x_new[i] < r_new[i] - 1e-9 or x_new[i] > w - r_new[i] + 1e-9:
                valid = False
                break
            if y_new[i] < r_new[i] - 1e-9 or y_new[i] > h - r_new[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x_new[i]-x_new[j])**2 + (y_new[i]-y_new[j])**2)
                    if d < r_new[i] + r_new[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            x, y, r = x_new, y_new, r_new
        else:
            break
    
    return x, y, r

def _gradient_optimize(x, y, r, w, h, deadline):
    """Use SLSQP with proper constraints to refine."""
    def unpack(vars_):
        return vars_[0::3], vars_[1::3], vars_[2::3]
    
    def objective(vars_):
        _, _, rr = unpack(vars_)
        return -np.sum(rr)
    
    def objective_grad(vars_):
        _, _, rr = unpack(vars_)
        g = np.zeros(3*N)
        g[2::3] = -1.0
        return g
    
    # Build constraint list
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i, w=w: w - v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i, h=h: h - v[3*i+1] - v[3*i+2]})
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j:
                         np.sqrt((v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2) - v[3*i+2] - v[3*j+2]})
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
    
    x0 = np.zeros(3*N)
    for i in range(N):
        x0[3*i] = x[i]
        x0[3*i+1] = y[i]
        x0[3*i+2] = r[i]
    
    try:
        result = minimize(
            objective, x0, jac=objective_grad,
            method='SLSQP', constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if result.fun < 0:
            rx, ry, rr = unpack(result.x)
            if _check_valid(rx, ry, rr, w, h):
                return rx, ry, rr, np.sum(rr)
    except Exception:
        pass
    
    return x, y, r, np.sum(r)

def _grid_init(n_cols, n_rows, w, h):
    """Generate staggered grid layout."""
    xs = np.zeros(N)
    ys = np.zeros(N)
    for i in range(N):
        col = i % n_cols
        row = i // n_cols
        x = (col + 0.5) * w / n_cols
        y = (row + 0.5) * h / n_rows
        if row % 2 == 1 and n_cols > 1:
            x += 0.3 * w / n_cols
            if x > w - 0.1*w/n_cols:
                x -= 0.6 * w / n_cols
        xs[i] = x
        ys[i] = y
    return xs, ys

def _random_init(w, h, seed):
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.05*w, 0.95*w, N)
    ys = rng.uniform(0.05*h, 0.95*h, N)
    return xs, ys

def _hilbert_init(w, h, seed):
    """Use a space-filling curve-like ordering for structured init."""
    rng = np.random.RandomState(seed)
    # Generate points in a grid-like pattern with jitter
    n_grid = 5
    points = []
    for i in range(n_grid * n_grid):
        gx = (i % n_grid) / n_grid
        gy = (i // n_grid) / n_grid
        points.append((gx, gy))
    # Shuffle and take first N
    rng.shuffle(points)
    xs = np.array([(p[0] + 0.05 + 0.9*rng.random()) * w / n_grid for p in points[:N]])
    ys = np.array([(p[1] + 0.05 + 0.9*rng.random()) * h / n_grid for p in points[:N]])
    return xs, ys

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_score = 0.0
    best_result = None
    
    # Aspect ratios to try: w + h = 2, so w = 2/(1+ar), h = 2*w*ar/(1+ar)
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.6, 0.62, 2.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Try multiple initialization strategies
        inits = []
        
        # Grid-based inits with different aspect-aware dimensions
        n_cols = max(1, int(round(np.sqrt(N * w / h))))
        n_rows = max(1, int(ceil(N / n_cols)))
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1
        xs_g, ys_g = _grid_init(n_cols, n_rows, w, h)
        inits.append((xs_g, ys_g))
        
        # Random inits
        for seed in [42, 123, 777]:
            if time.time() > deadline - 10:
                break
            xs_r, ys_r = _random_init(w, h, seed)
            inits.append((xs_r, ys_r))
        
        # Hilbert-like init
        xs_h, ys_h = _hilbert_init(w, h, 999)
        inits.append((xs_h, ys_h))
        
        for xs, ys in inits:
            if time.time() > deadline - 10:
                break
            
            # Compute initial radius from minimum spacing
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, xs[i], w-xs[i], ys[i], h-ys[i])
            
            r0 = 0.45 * min_dist
            radii = np.full(N, r0)
            
            # Force-directed relaxation with radius growth
            x_f, y_f, r_f = _force_directed_relax(xs, ys, radii, w, h, 
                                                     max_iter=1500, dt=0.003, deadline=deadline)
            score_f = np.sum(r_f)
            
            # Gradient refinement
            x_g, y_g, r_g, score_g = _gradient_optimize(x_f, y_f, r_f, w, h, deadline)
            
            if score_g > score_f:
                score = score_g
                x_best, y_best, r_best = x_g, y_g, r_g
            else:
                score = score_f
                x_best, y_best, r_best = x_f, y_f, r_f
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_best, y_best, r_best])
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i+0.5)/side, (j+0.5)/side, 0.5/side - 0.001])
        return np.array(cells[:N])
    
    return best_result

def ceil(x):
    import math
    return math.ceil(x)
# EVOLVE-BLOCK-END
