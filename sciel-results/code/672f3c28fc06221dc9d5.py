# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _force_directed_packing(w, h, n_iters=2000, seed=0):
    """
    Force-directed packing with radius growth.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize positions in a grid-like fashion with jitter
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Distribute centers
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols + rng.uniform(-0.1, 0.1) * (w/cols)
        y[i] = (row + 0.5) * h / rows + rng.uniform(-0.1, 0.1) * (h/rows)
        
        # Initial radius: small fraction of cell size
        r[i] = 0.3 * min(w/cols, h/rows)
        
        # Clamp positions
        x[i] = np.clip(x[i], r[i], w - r[i])
        y[i] = np.clip(y[i], r[i], h - r[i])

    # Force-directed relaxation
    for it in range(n_iters):
        t = it / n_iters
        lr_pos = 0.01 * (1.0 - t) + 0.001
        lr_rad = 0.005 * (1.0 - t) + 0.0005
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-10:
                    dx = rng.uniform(-1e-6, 1e-6)
                    dy = rng.uniform(-1e-6, 1e-6)
                    dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < min_dist:
                    # Overlap: repel
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                    # Shrink radii to help resolve overlap? Or just move?
                    # Moving is usually better for preserving sum of radii.
                elif dist < min_dist * 1.05:
                    # Near contact: weak spring to maintain contact
                    diff = dist - min_dist
                    force = 0.5 * diff
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary interactions and growth
        for i in range(N):
            # Check distance to walls
            d_left = x[i] - r[i]
            d_right = (w - x[i]) - r[i]
            d_bottom = y[i] - r[i]
            d_top = (h - y[i]) - r[i]
            
            min_wall_dist = min(d_left, d_right, d_bottom, d_top)
            
            # Check if touching any other circle
            touching_other = False
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] + 1e-5:
                    touching_other = True
                    break
            
            # Repulsion from walls
            if d_left < 0: fx[i] += 5.0 * abs(d_left)
            elif d_left < 0.05: fx[i] -= 0.5 * (0.05 - d_left) # Weak attraction to wall? No, just repulsion if inside
            # Actually, standard repulsion:
            if x[i] < r[i]:
                fx[i] += 10.0 * (r[i] - x[i])
            if x[i] > w - r[i]:
                fx[i] -= 10.0 * (x[i] - (w - r[i]))
            if y[i] < r[i]:
                fy[i] += 10.0 * (r[i] - y[i])
            if y[i] > h - r[i]:
                fy[i] -= 10.0 * (y[i] - (h - r[i]))
            
            # Growth force
            # If not touching anything, grow
            if not touching_other and min_wall_dist > 1e-4:
                fr[i] += 1.0
            # If touching a wall, don't grow (or shrink slightly if overlapping)
            # If touching another circle, don't grow
        
        # Update
        x += lr_pos * fx
        y += lr_pos * fy
        r += lr_rad * fr
        
        # Clamp positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
            # Ensure radius is positive
            r[i] = max(r[i], 1e-4)
            
    return x, y, r


def _optimize_radii(x, y, r, w, h):
    """
    Use scipy to maximize sum of radii given positions?
    Or jointly optimize x, y, r?
    Joint optimization is better.
    """
    def objective(vars):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:]
        
        # Penalty for violations
        penalty = 0.0
        
        # Boundary constraints
        for i in range(N):
            if x[i] < r[i]: penalty += (r[i] - x[i])**2
            if x[i] > w - r[i]: penalty += (x[i] - (w - r[i]))**2
            if y[i] < r[i]: penalty += (r[i] - y[i])**2
            if y[i] > h - r[i]: penalty += (y[i] - (h - r[i]))**2
            if r[i] < 0: penalty += r[i]**2
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(r) + 10.0 * penalty

    # Initial guess from force-directed
    x0 = np.concatenate([x, y, r])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w)) # x
    for i in range(N):
        bounds.append((0, h)) # y
    for i in range(N):
        bounds.append((1e-4, min(w, h)/2)) # r
        
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            # Validate
            if _validate_packing(x_opt, y_opt, r_opt, w, h):
                return x_opt, y_opt, r_opt
    except Exception:
        pass
        
    return x, y, r


def _validate_packing(x, y, r, w, h, tol=1e-6):
    """Check if packing is valid."""
    for i in range(len(x)):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    
    return True


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios
    # w + h = 2
    # Optimal aspect ratio is often close to 1 or related to N
    aspect_ratios = np.linspace(0.6, 1.4, 20)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing with a few seeds
        for seed in range(5):
            x, y, r = _force_directed_packing(w, h, n_iters=1000, seed=seed)
            
            # Polish with scipy
            x_opt, y_opt, r_opt = _optimize_radii(x, y, r, w, h)
            
            if _validate_packing(x_opt, y_opt, r_opt, w, h):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x_opt, y_opt, r_opt])
    
    if best_result is not None:
        return best_result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
