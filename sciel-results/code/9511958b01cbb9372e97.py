"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # Safety margin under 150s wall-clock


def _validate(x, w, h):
    """Check that all circles are valid: inside rectangle, non-overlapping."""
    n = len(x)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = x[i]
        if r <= 1e-12:
            return False
        if cx - r < -1e-9 or cx + r > w + 1e-9:
            return False
        if cy - r < -1e-9 or cy + r > h + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2
            if d2 < (x[i, 2] + x[j, 2]) ** 2 - 1e-9:
                return False
    return True


def _total_radius(x):
    return float(np.sum(x[:, 2]))


def _make_hex_grid(n, w, h):
    """Create a hexagonal grid of initial circles."""
    # Hexagonal packing is denser.
    # Approximate radius for hex packing of n circles in area w*h:
    # Area per circle ~ pi*r^2. Density ~ 0.9.
    # n * pi * r^2 ~ 0.9 * w * h => r ~ sqrt(0.9 * w * h / (n * pi))
    r_est = np.sqrt(0.9 * w * h / (n * np.pi))
    
    # Generate hex grid points
    points = []
    # Hex spacing
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    # Determine grid size
    cols = int(np.ceil(w / dx)) + 2
    rows = int(np.ceil(h / dy)) + 2
    
    # Start offset to center
    start_x = w / 2 - (cols - 1) * dx / 2
    start_y = h / 2 - (rows - 1) * dy / 2
    
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * dx
            y = start_y + row * dy
            if row % 2 == 1:
                x += dx / 2
            points.append([x, y])
            
    # Sort by distance to center to pick the best n
    cx_center, cy_center = w / 2, h / 2
    points.sort(key=lambda p: (p[0] - cx_center)**2 + (p[1] - cy_center)**2)
    
    # Take the first n points
    selected = points[:n]
    
    # Convert to circles with estimated radius
    circles = []
    for p in selected:
        # Ensure inside bounds
        cx = max(r_est, min(w - r_est, p[0]))
        cy = max(r_est, min(h - r_est, p[1]))
        circles.append([cx, cy, r_est * 0.8]) # Start slightly smaller to allow growth
        
    return np.array(circles)


def _solve_aspect(w, h, t0):
    """Solve packing for a given aspect ratio."""
    # Initial guess: Hex grid
    x0 = _make_hex_grid(N, w, h)
    
    # Flatten for optimizer
    x0_flat = x0.flatten()
    
    def objective(v):
        x = v.reshape(N, 3)
        # Maximize sum of radii => Minimize negative sum
        # Add penalty for violations to guide optimizer
        penalty = 0.0
        # Wall penalties
        for i in range(N):
            cx, cy, r = x[i]
            if cx - r < 0: penalty += (cx - r)**2
            if cx + r > w: penalty += (cx + r - w)**2
            if cy - r < 0: penalty += (cy - r)**2
            if cy + r > h: penalty += (cy + r - h)**2
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i,0] - x[j,0]
                dy = x[i,1] - x[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                need = x[i,2] + x[j,2]
                if d < need:
                    penalty += (need - d)**2
        return -np.sum(x[:, 2]) + 100.0 * penalty

    def objective_grad(v):
        # Gradient is hard with penalties, let COBYLA estimate or use SLSQP with constraints
        # For robustness, let's use SLSQP with explicit constraints but better start
        return None # Let SLSQP estimate if not provided? No, SLSQP requires jac or uses finite diff.
        # Finite diff is slow. Let's provide a simple gradient for the objective part and rely on constraints?
        # Actually, let's just use SLSQP with constraints and the hex start.

    # Define constraints for SLSQP
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2], 'jac': lambda v, i=i: _wall_jac(v, i, 0, w, h)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[3*i] - v[3*i+2], 'jac': lambda v, i=i: _wall_jac(v, i, 1, w, h)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2], 'jac': lambda v, i=i: _wall_jac(v, i, 2, w, h)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[3*i+1] - v[3*i+2], 'jac': lambda v, i=i: _wall_jac(v, i, 3, w, h)})

    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda v, i=i, j=j: _pair_dist(v, i, j) - v[3*i+2] - v[3*j+2],
                                'jac': lambda v, i=i, j=j: _pair_jac(v, i, j)})

    bounds = []
    for i in range(N):
        bounds.append((1e-6, w - 1e-6))   # cx
        bounds.append((1e-6, h - 1e-6))   # cy
        bounds.append((1e-6, min(w, h) / 2))  # r

    # Objective function for SLSQP
    def obj_slsqp(v):
        return -np.sum(v[2::3])

    def obj_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g

    x_best = x0_flat
    score_best = _total_radius(x0)
    
    # Run SLSQP
    if time.time() - t0 < DEADLINE:
        try:
            res = minimize(obj_slsqp, x0_flat, jac=obj_grad, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-6})
            if res.x is not None:
                x_res = res.x.reshape(N, 3)
                if _validate(x_res, w, h):
                    score_res = _total_radius(x_res)
                    if score_res > score_best:
                        x_best = x_res
                        score_best = score_res
                else:
                    # Try to repair by shrinking
                    x_repair = x_res.copy()
                    # Shrink radii until valid
                    for _ in range(10):
                        if _validate(x_repair, w, h):
                            break
                        x_repair[:, 2] *= 0.95
                    if _validate(x_repair, w, h):
                        score_res = _total_radius(x_repair)
                        if score_res > score_best:
                            x_best = x_repair
                            score_best = score_res
        except Exception:
            pass

    return x_best.reshape(N, 3)


def _wall_jac(v, i, wall_idx, w, h):
    g = np.zeros(len(v))
    if wall_idx == 0:  # cx - r >= 0
        g[3*i] = 1.0
        g[3*i+2] = -1.0
    elif wall_idx == 1:  # w - cx - r >= 0
        g[3*i] = -1.0
        g[3*i+2] = -1.0
    elif wall_idx == 2:  # cy - r >= 0
        g[3*i+1] = 1.0
        g[3*i+2] = -1.0
    elif wall_idx == 3:  # h - cy - r >= 0
        g[3*i+1] = -1.0
        g[3*i+2] = -1.0
    return g


def _pair_dist(v, i, j):
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    return np.sqrt(dx*dx + dy*dy + 1e-24)


def _pair_jac(v, i, j):
    g = np.zeros(len(v))
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    d = np.sqrt(dx*dx + dy*dy + 1e-24)
    g[3*i] = dx / d
    g[3*i+1] = dy / d
    g[3*j] = -dx / d
    g[3*j+1] = -dy / d
    g[3*i+2] = -1.0
    g[3*j+2] = -1.0
    return g


def construct():
    t0 = time.time()
    best = None
    best_score = -np.inf

    # Try a range of aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    # ar = w/h => w = ar*h => h(ar+1)=2 => h=2/(ar+1), w=2ar/(ar+1)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 0.75, 1.33, 0.65, 1.55]
    
    for ar in ratios:
        if time.time() - t0 > DEADLINE:
            break
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() - t0 > DEADLINE:
                break
            try:
                x = _solve_aspect(ww, hh, t0)
            except Exception:
                continue
            s = _total_radius(x)
            if _validate(x, ww, hh) and s > best_score:
                best_score = s
                best = (x.copy(), ww, hh)

    if best is None:
        # Fallback: Simple valid grid
        side = 5
        w = 2.0
        h = 2.0
        r = w / side * 0.9
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                cells.append([cx, cy, r])
        return np.array(cells[:N])

    x, w, h = best
    # Final safety check
    if not _validate(x, w, h):
        # Shrink to ensure validity
        for _ in range(10):
            if _validate(x, w, h):
                break
            x[:, 2] *= 0.95
        if not _validate(x, w, h):
            # Ultimate fallback
            side = 5
            w = 2.0
            h = 2.0
            r = w / side * 0.9
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= N:
                        break
                    cx = (i + 0.5) * w / side
                    cy = (j + 0.5) * h / side
                    cells.append([cx, cy, r])
            return np.array(cells[:N])
    return x
