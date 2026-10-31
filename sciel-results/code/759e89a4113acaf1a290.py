"""Rewritten program: circle packing in perimeter-4 rectangle, 21 circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 115.0  # safety margin under 120s


def _perimeter_constraint(x, W, H):
    """Return (2W+2H - 4) which must be <= 0."""
    return 2 * W + 2 * H - 4.0


def _build_objective_and_constraints(W, H, n):
    """
    Build objective (maximize sum of radii -> minimize negative sum)
    and constraints (non-overlap, boundary).
    Variables: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    """
    def objective(v):
        radii = v[2::3]
        return -np.sum(radii)

    # Inequality constraints: g_i >= 0
    # 1) r_i >= 0  ->  r_i >= 0  (lower bound handled by bounds)
    # 2) x_i - r_i >= 0
    # 3) x_i + r_i <= W  ->  W - x_i - r_i >= 0
    # 4) y_i - r_i >= 0
    # 5) y_i + r_i <= H  ->  H - y_i - r_i >= 0
    # 6) (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2  ->  (x_i-x_j)^2+(y_i-y_j)^2-(r_i+r_j)^2 >= 0

    def make_boundary_constraints():
        cons = []
        for i in range(n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            cons.append({'type': 'ineq', 'fun': lambda v, xi=xi: v[xi] - v[xi+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, xi=xi: W - v[xi] - v[xi+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, yi=yi: v[yi] - v[yi+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, yi=yi: H - v[yi] - v[yi+2]})
        return cons

    def make_overlap_constraints():
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = 3*i, 3*i+1, 3*i+2
                xj, yj, rj = 3*j, 3*j+1, 3*j+2
                def overlap(v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                    dx = v[xi] - v[xj]
                    dy = v[yi] - v[yj]
                    return dx*dx + dy*dy - (v[ri]+v[rj])**2
                cons.append({'type': 'ineq', 'fun': overlap})
        return cons

    constraints = make_boundary_constraints() + make_overlap_constraints()
    return objective, constraints


def _make_initial_guess(W, H, n, grid_res=6):
    """Create a grid-based initial guess."""
    # Place circles on a grid, with small radii
    cols = int(np.ceil(np.sqrt(n * W / H))) if H > 0 else grid_res
    rows = int(np.ceil(n / cols))
    
    xs = []
    ys = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = (c + 0.5) * W / cols
        y = (r + 0.5) * H / rows
        xs.append(x)
        ys.append(y)
    
    # Initial radius: small fraction of min(W,H)/max(cols,rows)
    base_r = 0.4 * min(W/cols, H/rows)
    radii = [base_r] * n
    
    v0 = np.zeros(3*n)
    for i in range(n):
        v0[3*i] = xs[i]
        v0[3*i+1] = ys[i]
        v0[3*i+2] = radii[i]
    
    return v0


def _optimize(W, H, v0, max_iter=5000, tol=1e-8):
    """Run SLSQP optimization."""
    n = len(v0) // 3
    objective, constraints = _build_objective_and_constraints(W, H, n)
    
    # Bounds: x in [0, W], y in [0, H], r in [0, min(W,H)/2]
    r_max = 0.5 * min(W, H)
    bounds = []
    for i in range(n):
        bounds.append((0, W))
        bounds.append((0, H))
        bounds.append((1e-6, r_max))
    
    res = minimize(
        objective, v0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': max_iter, 'ftol': tol}
    )
    return res


def _validate_and_compute(W, H, v):
    """Validate the solution and compute sum of radii."""
    n = len(v) // 3
    radii = v[2::3]
    
    # Check radii positive
    if np.any(radii < 1e-8):
        return None, 0.0
    
    # Check boundary
    for i in range(n):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        if x - r < -1e-6 or x + r > W + 1e-6 or y - r < -1e-6 or y + r > H + 1e-6:
            return None, 0.0
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist_sq = dx*dx + dy*dy
            min_dist = v[3*i+2] + v[3*j+2]
            if dist_sq < min_dist*min_dist - 1e-6:
                return None, 0.0
    
    # Check perimeter
    perim = 2*W + 2*H
    if perim > 4.0 + 1e-6:
        return None, 0.0
    
    return v, np.sum(radii)


def _try_aspect_ratio(W, H, time_start, best_val, best_v, best_WH):
    """Try to optimize for given W,H. Return updated bests."""
    n = N
    v0 = _make_initial_guess(W, H, n)
    
    try:
        res = _optimize(W, H, v0, max_iter=3000, tol=1e-8)
        if res.success:
            v = res.x
            # Clamp to feasible region
            for i in range(n):
                v[3*i] = np.clip(v[3*i], 0, W)
                v[3*i+1] = np.clip(v[3*i+1], 0, H)
                v[3*i+2] = max(1e-6, v[3*i+2])
            
            v_valid, val = _validate_and_compute(W, H, v)
            if v_valid is not None and val > best_val:
                return val, v_valid, (W, H)
    except Exception:
        pass
    
    return best_val, best_v, best_WH


def construct():
    start_time = time.time()
    best_val = 0.0
    best_v = None
    best_WH = (1.0, 1.0)
    
    # Try a range of aspect ratios
    # Perimeter = 2(W+H) = 4 -> W+H = 2
    # W in [0.3, 1.7] (H = 2-W)
    aspect_ratios = []
    for W in np.linspace(0.3, 1.7, 25):
        H = 2.0 - W
        if H > 0.1:
            aspect_ratios.append((W, H))
    
    # Also try some specific aspect ratios
    for W in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        H = 2.0 - W
        if H > 0.1 and (W, H) not in aspect_ratios:
            aspect_ratios.append((W, H))
    
    # Sort by likely promise: start near square
    aspect_ratios.sort(key=lambda wh: abs(wh[0] - wh[1]))
    
    for W, H in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:  # Leave 10s margin
            break
        
        best_val, best_v, best_WH = _try_aspect_ratio(W, H, start_time, best_val, best_v, best_WH)
    
    # If we still have time, do a polish pass on the best solution
    if best_v is not None and time.time() - start_time < TIME_LIMIT - 20:
        W, H = best_WH
        n = N
        objective, constraints = _build_objective_and_constraints(W, H, n)
        r_max = 0.5 * min(W, H)
        bounds = []
        for i in range(n):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                objective, best_v, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-10}
            )
            if res.success:
                v = res.x
                for i in range(n):
                    v[3*i] = np.clip(v[3*i], 0, W)
                    v[3*i+1] = np.clip(v[3*i+1], 0, H)
                    v[3*i+2] = max(1e-6, v[3*i+2])
                
                v_valid, val = _validate_and_compute(W, H, v)
                if v_valid is not None and val > best_val:
                    best_val = val
                    best_v = v_valid
        except Exception:
            pass
    
    # Format output: (x, y, r) for each circle
    if best_v is not None:
        out = np.zeros((N, 3))
        for i in range(N):
            out[i] = best_v[3*i:3*i+3]
        return out
    else:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(N):
            c = i % side
            r = i // side
            x = (c + 0.5) / side
            y = (r + 0.5) / side
            rad = 0.5 / side - 1e-3
            cells.append([x, y, rad])
        return np.array(cells)


# EVOLVE-BLOCK-END
