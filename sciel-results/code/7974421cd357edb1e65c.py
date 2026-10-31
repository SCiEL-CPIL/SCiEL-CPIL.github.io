"""Circle packing in perimeter-4 rectangle: DE for topology + SLSQP for refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # Safety margin before 150s kill

def _max_feasible_radii(centers, W, H, max_iter=100, tol=1e-8):
    """
    Given centers and box dimensions, compute the maximum radius for each circle
    such that all circles are disjoint and inside the box.
    Uses iterative relaxation to find consistent radii.
    Returns an array of radii or None if infeasible.
    """
    n = len(centers)
    # Initialize radii based on wall distances
    radii = np.zeros(n)
    for i in range(n):
        cx, cy = centers[i]
        radii[i] = min(cx, W - cx, cy, H - cy)
        if radii[i] <= 0:
            return None
    
    # Iteratively adjust radii to satisfy non-overlap constraints
    for _ in range(max_iter):
        changed = False
        new_radii = radii.copy()
        for i in range(n):
            # Start with wall constraint
            r_i = new_radii[i]
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = np.hypot(dx, dy)
                # r_i + r_j <= d => r_i <= d - r_j
                max_r_i = d - new_radii[j]
                if max_r_i < r_i:
                    r_i = max_r_i
                    changed = True
            # Also ensure wall constraint
            cx, cy = centers[i]
            wall_r = min(cx, W - cx, cy, H - cy)
            if wall_r < r_i:
                r_i = wall_r
                changed = True
            new_radii[i] = max(r_i, 0.0)
        
        if not changed or np.max(np.abs(new_radii - radii)) < tol:
            radii = new_radii
            break
        radii = new_radii
    
    # Final check: all radii must be positive
    if np.any(radii <= 1e-10):
        return None
    
    # Verify no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d + 1e-9 < radii[i] + radii[j]:
                return None
    
    return radii

def _de_objective(x):
    """
    Objective for Differential Evolution.
    x[0] = W (width), x[1:] = centers (2N values)
    Returns negative sum of max feasible radii (using d/2 heuristic).
    """
    W = x[0]
    H = 2.0 - W
    if W < 0.2 or H < 0.2:
        return 1e6
    
    centers = x[1:].reshape(N, 2)
    
    # Quick check for centers inside box
    if np.any(centers < 0) or np.any(centers[:, 0] > W) or np.any(centers[:, 1] > H):
        return 1e6
        
    radii = _max_feasible_radii(centers, W, H)
    if radii is None:
        return 1e6
        
    return -np.sum(radii)

def _slsqp_objective(x, W, H, mu=1e3):
    """
    Objective for SLSQP.
    x is shape (N,3): cx, cy, r.
    Returns (value, grad).
    Value = -sum(r) + penalties for violations.
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)

    # Clip for numerical stability in gradient calculation
    cx_c = np.clip(cx, 0.0, W)
    cy_c = np.clip(cy, 0.0, H)
    r_c = np.clip(r, 1e-6, 1.0)

    val = -np.sum(r_c)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary constraints: r <= min(cx, cy, W-cx, H-cy)
    lo = np.minimum(np.minimum(cx_c, cy_c), np.minimum(W - cx_c, H - cy_c))
    viol_b = np.maximum(r_c - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
    
    for i in range(n):
        if viol_b[i] > 0:
            d_left = cx_c[i]
            d_bottom = cy_c[i]
            d_right = W - cx_c[i]
            d_top = H - cy_c[i]
            min_d = lo[i]
            eps = 1e-9
            g_cx = 0.0
            g_cy = 0.0
            cnt = 0
            if d_left <= min_d + eps:
                g_cx += 1.0; cnt += 1
            if d_right <= min_d + eps:
                g_cx += -1.0; cnt += 1
            if d_bottom <= min_d + eps:
                g_cy += 1.0; cnt += 1
            if d_top <= min_d + eps:
                g_cy += -1.0; cnt += 1
            if cnt > 0:
                g_cx /= cnt
                g_cy /= cnt
            dv = 2.0 * mu * viol_b[i]
            grad[i, 0] += dv * (-g_cx)
            grad[i, 1] += dv * (-g_cy)
            grad[i, 2] += dv * 1.0

    # Non-overlap: max(r_i + r_j - dist, 0)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_c[i] - cx_c[j]
            dy = cy_c[i] - cy_c[j]
            d = np.hypot(dx, dy)
            req = r_c[i] + r_c[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu * viol * 1.0
                grad[j, 0] += 2.0 * mu * viol * (-1.0)
                grad[i, 2] += 2.0 * mu * viol * 1.0
                grad[j, 2] += 2.0 * mu * viol * 1.0
                continue
            dv = 2.0 * mu * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad

def _evaluate(x, W, H):
    """Return sum of radii if valid, else None."""
    cx = np.clip(x[:, 0], 0.0, W)
    cy = np.clip(x[:, 1], 0.0, H)
    r = np.maximum(x[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _grid_layout(kx, ky, W, H, shrink=0.999):
    """Build a kx*ky grid layout."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _run_de(bounds, pop_size=30, max_iter=200, time_limit=60.0):
    """Simple Differential Evolution implementation."""
    n_dims = len(bounds)
    start_time = time.time()
    
    # Initialize population
    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])
    
    fitness = np.array([_de_objective(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        new_pop = []
        new_fitness = []
        
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            F = 0.5 + random.random() * 0.5
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < 0.7:
                    trial[j] = mutant[j]
            
            f_trial = _de_objective(trial)
            
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
        
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = pop[gen_best_idx].copy()
            
    return best_x, best_f

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    candidates = []
    
    # Generate multiple grid-based candidates with proper radius calculation
    grid_configs = [
        (3, 7, 1.0), (4, 6, 1.0), (5, 5, 1.0), (6, 4, 1.0), (7, 3, 1.0),
        (3, 7, 0.9), (4, 6, 0.9), (5, 5, 0.9), (6, 4, 0.9), (7, 3, 0.9),
        (3, 7, 1.1), (4, 6, 1.1), (5, 5, 1.1), (6, 4, 1.1), (7, 3, 1.1),
        (3, 7, 0.8), (4, 6, 0.8), (5, 5, 0.8), (6, 4, 0.8), (7, 3, 0.8),
    ]
    
    for kx, ky, W in grid_configs:
        if kx * ky < N:
            continue
        H = 2.0 - W
        if H < 0.3:
            continue
        g = _grid_layout(kx, ky, W, H, shrink=0.98)
        s = _evaluate(g, W, H)
        if s is not None:
            candidates.append((s, W, H, g))
    
    # Add some random initializations
    for trial in range(20):
        W = random.uniform(0.7, 1.3)
        H = 2.0 - W
        centers = np.random.uniform(0.05, W - 0.05, (N, 2))
        centers[:, 1] = np.random.uniform(0.05, H - 0.05, N)
        radii = _max_feasible_radii(centers, W, H)
        if radii is not None:
            x0 = np.column_stack([centers, radii])
            s = _evaluate(x0, W, H)
            if s is not None:
                candidates.append((s, W, H, x0))
    
    # Sort candidates by score
    candidates.sort(key=lambda t: -t[0])
    
    # Refine top candidates with SLSQP
    top_k = min(10, len(candidates))
    
    for idx in range(top_k):
        if time.time() > DEADLINE - 15:
            break
            
        s0, W, H, g0 = candidates[idx]
        
        def fun(x):
            v, _ = _slsqp_objective(x, W, H)
            return v
            
        def grad(x):
            _, g = _slsqp_objective(x, W, H)
            return g
            
        try:
            res = minimize(fun, g0, jac=grad, method='SLSQP',
                           options={'maxiter': 80, 'ftol': 1e-9, 'disp': 0})
            if res.x is not None:
                s = _evaluate(res.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_arr = res.x.copy()
                    best_W = W
                    best_H = H
        except Exception:
            pass
    
    # Use best candidate if SLSQP didn't improve
    if best_arr is None and len(candidates) > 0:
        best_sum = candidates[0][0]
        best_arr = candidates[0][3].copy()
        best_W = candidates[0][1]
        best_H = candidates[0][2]
    
    # Final processing
    if best_arr is None:
        # Ultimate fallback
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.95)
        best_arr = g
        best_W, best_H = W, H
    
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    centers = np.column_stack([cx, cy])
    
    radii = _max_feasible_radii(centers, best_W, best_H)
    if radii is None:
        radii = np.zeros(N)
        for i in range(N):
            radii[i] = min(cx[i], best_W - cx[i], cy[i], best_H - cy[i]) * 0.45
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j]) * 0.999
                    radii[i] *= scale
                    radii[j] *= scale
        radii = np.maximum(radii, 1e-6)
    
    out = np.column_stack([cx, cy, radii])
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        W, H = 1.0, 1.0
        cx_g = np.linspace(0.1, 0.9, 5)
        cy_g = np.linspace(0.1, 0.9, 5)
        pts = []
        for yy in cy_g:
            for xx in cx_g:
                r = min(xx, yy, W - xx, H - yy) * 0.9
                pts.append([xx, yy, r])
        arr = np.array(pts[:N])
        return arr
    
    return out
