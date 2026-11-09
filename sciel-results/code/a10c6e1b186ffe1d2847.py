"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # safety margin before 120s kill

def _perimeter_constraint(W, H):
    """2*(W+H) <= 4  =>  W+H <= 2"""
    return 2.0 - (W + H)

def _objective(x, W, H):
    """Maximize sum of radii => minimize -sum(r).
    x layout: [W, H, (cx1,cy1,r1), (cx2,cy2,r2), ...]
    """
    W, H = x[0], x[1]
    radii = x[2::3]
    return -np.sum(radii)

def _constraints(W, H):
    """Return list of constraint dicts for scipy minimize (SLSQP)."""
    cons = []
    # Perimeter: 2*(W+H) <= 4
    cons.append({'type': 'ineq', 'fun': lambda x: 2.0 - (x[0] + x[1])})
    # W > 0, H > 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.01})
    cons.append({'type': 'ineq', 'fun': lambda x: x[1] - 0.01})
    # For each circle i: cx_i - r_i >= 0, cx_i + r_i <= W, cy_i - r_i >= 0, cy_i + r_i <= H
    for i in range(N):
        base = 2 + 3*i
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[b] - x[b+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[0] - x[b] - x[b+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[b+1] - x[b+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[1] - x[b+1] - x[b+2]})
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i+1, N):
            bi = 2 + 3*i
            bj = 2 + 3*j
            cons.append({
                'type': 'ineq',
                'fun': lambda x, a=bi, b=bj:
                    (x[a]-x[b])**2 + (x[a+1]-x[b+1])**2 - (x[a+2]+x[b+2])**2
            })
    return cons

def _extract(x):
    """Extract (cx, cy, r) array of shape (N,3) from x."""
    W, H = x[0], x[1]
    pts = x[2:].reshape(N, 3)
    return pts, W, H

def _is_valid(pts, W, H, tol=1e-6):
    """Check all constraints are satisfied."""
    if W <= 0 or H <= 0:
        return False
    if 2*(W+H) > 4.0 + tol:
        return False
    for i in range(N):
        cx, cy, r = pts[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > W + tol or cy - r < -tol or cy + r > H + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d2 = (pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2
            if d2 < (pts[i,2]+pts[j,2])**2 - tol:
                return False
    return True

def _force_directed_relax(pts, W, H, iterations=200, dt=0.005):
    """Simple force-directed relaxation to push circles apart and grow radii."""
    pts = pts.copy().astype(float)
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        # Wall repulsion: push circles inward if too close to walls
        for i in range(N):
            cx, cy, r = pts[i]
            if cx - r < 0:
                forces[i, 0] += (r - cx) * 5.0
            if cx + r > W:
                forces[i, 0] -= (cx + r - W) * 5.0
            if cy - r < 0:
                forces[i, 1] += (r - cy) * 5.0
            if cy + r > H:
                forces[i, 1] -= (cy + r - H) * 5.0
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = pts[i,2] + pts[j,2]
                if d < 1e-9:
                    dx, dy = 0.01, 0.01
                    d = np.sqrt(dx*dx + dy*dy)
                if d < min_d:
                    overlap = (min_d - d) / d
                    fx = dx * overlap * 3.0
                    fy = dy * overlap * 3.0
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        # Grow radii slightly
        for i in range(N):
            forces[i, 2] += 0.001
        pts += dt * forces
        # Clamp positions to stay in bounds
        for i in range(N):
            pts[i, 0] = np.clip(pts[i, 0], pts[i, 2], W - pts[i, 2])
            pts[i, 1] = np.clip(pts[i, 1], pts[i, 2], H - pts[i, 2])
            pts[i, 2] = max(pts[i, 2], 0.01)
    return pts

def _slsqp_polish(pts, W, H, time_budget=10.0):
    """Polish with SLSQP. Returns improved (pts, W, H) or original if fails."""
    x0 = np.concatenate([[W, H], pts.ravel()])
    cons = _constraints(W, H)
    bounds = [(0.01, 2.0), (0.01, 2.0)]
    for i in range(N):
        bounds += [(0.0, 2.0), (0.0, 2.0), (0.01, 1.0)]
    
    t0 = time.time()
    try:
        res = minimize(
            _objective, x0, args=(W, H),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        if res.success or res.fun < _objective(x0, W, H):
            x = res.x
            pts_new, W_new, H_new = _extract(x)
            if _is_valid(pts_new, W_new, H_new):
                return pts_new, W_new, H_new
    except Exception:
        pass
    return pts, W, H

def _grid_init(side, W, H):
    """Grid initialization."""
    pts = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side * W
            cy = (j + 0.5) / side * H
            r = min(W, H) / (2 * side) * 0.9
            pts.append([cx, cy, r])
    return np.array(pts[:N])

def _random_init(W, H, rng):
    """Random initialization with small circles."""
    pts = []
    for _ in range(N):
        cx = rng.uniform(0.1, W - 0.1)
        cy = rng.uniform(0.1, H - 0.1)
        r = rng.uniform(0.03, 0.08)
        pts.append([cx, cy, r])
    return np.array(pts)

def _hex_init(W, H):
    """Hexagonal packing initialization."""
    pts = []
    rows = 5
    r0 = min(W, H) / (2 * rows) * 0.85
    for row in range(rows):
        n_in_row = N - len(pts)
        for col in range(n_in_row):
            cx = (col + 0.5) / max(n_in_row, 1) * W
            cy = (row + 0.5) / rows * H
            r = r0 * 0.9
            pts.append([cx, cy, r])
    return np.array(pts[:N])

def construct():
    t_start = time.time()
    best_score = -np.inf
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.5, 0.7, 2.0, 0.5, 1.2, 0.8]
    
    for ar in aspect_ratios:
        if time.time() - t_start > DEADLINE - 15:
            break
        # W + H = 2, W/H = ar
        # W = ar * H, ar*H + H = 2 => H = 2/(ar+1), W = 2*ar/(ar+1)
        H = 2.0 / (ar + 1.0)
        W = 2.0 * ar / (ar + 1.0)
        
        # Multiple initializations
        inits = []
        side = int(np.ceil(np.sqrt(N)))
        inits.append(_grid_init(side, W, H))
        inits.append(_hex_init(W, H))
        rng = np.random.RandomState(42)
        for seed in range(3):
            inits.append(_random_init(W, H, rng))
        
        for pts in inits:
            if time.time() - t_start > DEADLINE - 10:
                break
            # Force-directed relaxation
            pts_relaxed = _force_directed_relax(pts, W, H, iterations=300, dt=0.003)
            
            # SLSQP polish
            t_polish_start = time.time()
            budget = min(10.0, DEADLINE - 5 - (time.time() - t_start))
            if budget < 2.0:
                break
            pts_p, W_p, H_p = _slsqp_polish(pts_relaxed, W, H, time_budget=budget)
            
            if _is_valid(pts_p, W_p, H_p):
                score = np.sum(pts_p[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([pts_p[:, 0], pts_p[:, 1], pts_p[:, 2]])
    
    # Fallback to seed if nothing valid found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
