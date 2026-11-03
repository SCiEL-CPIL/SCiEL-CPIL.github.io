"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

_DEADLINE = 130.0  # seconds wall-clock safety margin


def _feasibility(x, W):
    """Compute feasibility violation for a given packing."""
    H = 2.0 - W
    n = N
    coords = x.reshape(n, 3)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rs = coords[:, 2]

    # Boundary violations
    viol_left = np.maximum(0.0, rs - xs)
    viol_right = np.maximum(0.0, xs + rs - W)
    viol_bottom = np.maximum(0.0, rs - ys)
    viol_top = np.maximum(0.0, ys + rs - H)
    boundary_viol = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)

    # Pairwise overlap violations
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            req = rs[i] + rs[j]
            if dist_sq < req * req:
                dist = np.sqrt(dist_sq)
                overlap_viol += (req - dist)

    return boundary_viol + overlap_viol


def _sum_radii(x):
    """Sum of all radii."""
    return np.sum(x[2::3])


def _objective_full(params):
    """
    Full objective: params = [W, x0, y0, r0, x1, y1, r1, ...]
    W is the width, rest are circle parameters.
    """
    W = params[0]
    x = params[1:]
    H = 2.0 - W
    
    # Ensure W is in valid range
    if W < 0.1 or W > 1.9:
        return 1e10
    
    n = N
    coords = x.reshape(n, 3)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rs = coords[:, 2]
    
    # Boundary violations
    viol_left = np.maximum(0.0, rs - xs)
    viol_right = np.maximum(0.0, xs + rs - W)
    viol_bottom = np.maximum(0.0, rs - ys)
    viol_top = np.maximum(0.0, ys + rs - H)
    boundary_viol = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)
    
    # Pairwise overlap violations
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            req = rs[i] + rs[j]
            if dist_sq < req * req:
                dist = np.sqrt(dist_sq)
                overlap_viol += (req - dist)
    
    feas = boundary_viol + overlap_viol
    sum_r = np.sum(rs)
    
    # Heavy penalty for infeasibility
    if feas > 1e-6:
        return 1e6 * feas - sum_r
    return -sum_r


def _grid_init(W, H, rng, jitter=0.15):
    """Initialize circles in a grid pattern with jitter."""
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros(3 * N)
    for i in range(N):
        row = i // side
        col = i % side
        cx = (col + 0.5) * W / side + rng.uniform(-jitter, jitter) * W / side
        cy = (row + 0.5) * H / side + rng.uniform(-jitter, jitter) * H / side
        r = 0.45 * min(W / side, H / side)
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        x0[3 * i] = cx
        x0[3 * i + 1] = cy
        x0[3 * i + 2] = r
    return x0


def _random_init(W, H, rng):
    """Random initialization with small radii."""
    x0 = np.zeros(3 * N)
    max_r = 0.5 * min(W, H) / np.sqrt(N)
    for i in range(N):
        r = max_r * rng.uniform(0.3, 0.7)
        cx = rng.uniform(r, W - r)
        cy = rng.uniform(r, H - r)
        x0[3 * i] = cx
        x0[3 * i + 1] = cy
        x0[3 * i + 2] = r
    return x0


def _local_optimize(x, W, max_time=5.0):
    """
    Local optimization for fixed W using L-BFGS-B with penalty.
    Returns (sum_r, x_optimized) or (None, None) if fails.
    """
    H = 2.0 - W
    start = time.time()
    current_x = x.copy()
    best_sum_r = -1e10
    best_x = current_x.copy()
    
    # Check initial feasibility
    feas = _feasibility(current_x, W)
    if feas < 1e-6:
        best_sum_r = _sum_radii(current_x)
    
    # Penalty-based optimization
    penalty_weights = [50.0, 200.0, 1000.0, 5000.0, 20000.0, 100000.0]
    
    for pw in penalty_weights:
        if time.time() - start > max_time - 1.0:
            break
        
        def obj_func(x):
            H_local = 2.0 - W
            n = N
            coords = x.reshape(n, 3)
            xs = coords[:, 0]
            ys = coords[:, 1]
            rs = coords[:, 2]
            
            # Objective
            obj = -np.sum(rs)
            
            # Violations
            viol_left = np.maximum(0.0, rs - xs)
            viol_right = np.maximum(0.0, xs + rs - W)
            viol_bottom = np.maximum(0.0, rs - ys)
            viol_top = np.maximum(0.0, ys + rs - H_local)
            boundary_viol = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)
            
            overlap_viol = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx * dx + dy * dy
                    req = rs[i] + rs[j]
                    if dist_sq < req * req:
                        dist = np.sqrt(dist_sq)
                        overlap_viol += (req - dist)
            
            feas = boundary_viol + overlap_viol
            return obj + pw * feas
        
        try:
            res = minimize(
                obj_func,
                current_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            current_x = res.x
            
            feas = _feasibility(current_x, W)
            if feas < 1e-5:
                sum_r = _sum_radii(current_x)
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_x = current_x.copy()
        except Exception:
            continue
    
    return best_sum_r, best_x


def _optimize_aspect_and_positions(x_full, max_time=8.0):
    """
    Optimize both aspect ratio W and circle positions simultaneously.
    x_full = [W, x0, y0, r0, ...]
    """
    start = time.time()
    current = x_full.copy()
    best_obj = _objective_full(current)
    best_x = current.copy()
    
    # Bounds: W in [0.3, 1.7], positions and radii positive
    n = N
    bounds = [(0.3, 1.7)]
    for i in range(n):
        bounds.append((0.01, 1.99))  # x
        bounds.append((0.01, 1.99))  # y
        bounds.append((0.01, 0.9))   # r
    
    # Use SLSQP for simultaneous optimization
    try:
        res = minimize(
            _objective_full,
            current,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10},
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # If we have time, do another round with tighter bounds
    if time.time() - start < max_time - 2.0:
        W_opt = best_x[0]
        x_opt = best_x[1:].copy()
        # Fix W and optimize positions only
        sum_r, x_refined = _local_optimize(x_opt, W_opt, max_time=3.0)
        if sum_r is not None and sum_r > _sum_radii(best_x[1:]):
            best_x = np.concatenate([[W_opt], x_refined])
            best_obj = _objective_full(best_x)
    
    return best_obj, best_x


def _differential_evolution_search(max_time=15.0):
    """
    Global search using differential evolution over [W, all circle params].
    """
    start = time.time()
    n = N
    
    # Bounds
    bounds = [(0.4, 1.6)]  # W
    for i in range(n):
        bounds.append((0.05, 1.95))  # x
        bounds.append((0.05, 1.95))  # y
        bounds.append((0.05, 0.8))   # r
    
    # Use a small population for speed
    try:
        res = differential_evolution(
            _objective_full,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=True,
            update_interval=1,
            callback=None,
            workers=1,
        )
        if time.time() - start < max_time:
            return res.fun, res.x
    except Exception:
        pass
    
    return None, None


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_sum_r = 0.0
    best_x_full = None
    
    # Strategy 1: Grid initializations with local optimization at several widths
    widths_to_try = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W in widths_to_try:
        if time.time() > _DEADLINE - 20.0:
            break
        H = 2.0 - W
        
        # Multiple initializations
        for init_type in ['grid', 'random']:
            if time.time() > _DEADLINE - 15.0:
                break
            
            rng = np.random.default_rng(seed=int(W * 1000) + (0 if init_type == 'grid' else 100))
            
            if init_type == 'grid':
                x0 = _grid_init(W, H, rng, jitter=0.15)
            else:
                x0 = _random_init(W, H, rng)
            
            sum_r, x_opt = _local_optimize(x0, W, max_time=4.0)
            
            if sum_r is not None and sum_r > best_sum_r:
                best_sum_r = sum_r
                best_x_full = np.concatenate([[W], x_opt])
    
    # Strategy 2: Simultaneous aspect + position optimization from best found
    if best_x_full is not None and time.time() < _DEADLINE - 25.0:
        obj, x_full_opt = _optimize_aspect_and_positions(best_x_full, max_time=10.0)
        if obj is not None:
            feas = _feasibility(x_full_opt[1:], x_full_opt[0])
            if feas < 1e-5:
                sum_r = _sum_radii(x_full_opt[1:])
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_x_full = x_full_opt.copy()
    
    # Strategy 3: Differential evolution global search if time permits
    if time.time() < _DEADLINE - 30.0:
        de_obj, de_x = _differential_evolution_search(max_time=20.0)
        if de_x is not None:
            feas = _feasibility(de_x[1:], de_x[0])
            if feas < 1e-5:
                sum_r = _sum_radii(de_x[1:])
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_x_full = de_x.copy()
    
    # Strategy 4: Final refinement of best solution
    if best_x_full is not None and time.time() < _DEADLINE - 10.0:
        W_best = best_x_full[0]
        x_best = best_x_full[1:]
        sum_r_final, x_final = _local_optimize(x_best, W_best, max_time=8.0)
        if sum_r_final is not None and sum_r_final > best_sum_r:
            best_sum_r = sum_r_final
            best_x_full = np.concatenate([[W_best], x_final])
    
    # Fallback: if no good solution, use simple grid
    if best_x_full is None:
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            r = 0.45 / side
            cx = (col + 0.5) / side * W
            cy = (row + 0.5) / side * H
            cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    # Extract and validate final solution
    W_final = best_x_full[0]
    H_final = 2.0 - W_final
    coords = best_x_full[1:].reshape(N, 3).copy()
    
    # Clamp to ensure feasibility
    for i in range(N):
        coords[i, 0] = np.clip(coords[i, 0], coords[i, 2], W_final - coords[i, 2])
        coords[i, 1] = np.clip(coords[i, 1], coords[i, 2], H_final - coords[i, 2])
    
    # Final feasibility check
    feas = _feasibility(coords.ravel(), W_final)
    if feas > 1e-3:
        # Fallback to grid
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            r = 0.45 / side
            cx = (col + 0.5) / side * W
            cy = (row + 0.5) / side * H
            cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    return coords
# EVOLVE-BLOCK-END
