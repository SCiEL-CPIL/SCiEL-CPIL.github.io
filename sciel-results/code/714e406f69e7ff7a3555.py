import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
_DEADLINE = 135.0  # Safety margin for 150s limit

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _penalty_and_grad(x, n=N, mu=10000.0):
    """
    x: (3n,) vector [x0, y0, r0, x1, y1, r1, ...]
    Minimize -sum(r) + mu * (boundary_violations + overlap_violations)
    """
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    obj = -np.sum(rs)
    
    # Boundary Violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)
    
    # Overlap Violations
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)
    
    total_penalty = bpen + open_
    energy = obj + mu * total_penalty
    
    # Gradient
    grad = np.zeros_like(x)
    grad[2::3] = -1.0
    
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))
    
    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri
    
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist
        
        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)
        
    return energy, grad

def _smooth_objective(x, n=N, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for min/max constraints.
    This makes the objective smoother and helps escape local minima.
    """
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Soft boundary constraints using soft-max
    # We want r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # Use soft-min: -logsumexp(-a/beta, -b/beta, ...) * beta
    
    def soft_min(vals, beta=50.0):
        # Returns smooth approximation of min(vals)
        m = np.min(vals)
        return m - np.log(np.mean(np.exp((vals - m) / beta))) * beta
    
    # For each circle, compute soft constraint violations
    # Boundary: r_i should be <= min(x_i, 1-x_i, y_i, 1-y_i)
    # Violation = max(0, r_i - min_dist_to_boundary)
    
    dist_to_left = xs
    dist_to_right = 1.0 - xs
    dist_to_bot = ys
    dist_to_top = 1.0 - ys
    
    # Soft min of distances to boundaries
    min_dist = np.zeros(n)
    for i in range(n):
        vals = np.array([dist_to_left[i], dist_to_right[i], dist_to_bot[i], dist_to_top[i]])
        min_dist[i] = soft_min(vals, beta)
    
    # Boundary violation: max(0, r_i - min_dist_i)
    # Use smooth max: logsumexp(0, (r-min)/beta) * beta
    bviol = np.zeros(n)
    for i in range(n):
        diff = rs[i] - min_dist[i]
        bviol[i] = np.log(np.exp(0) + np.exp(diff / beta)) * beta
    
    bpen = np.sum(bviol ** 2)
    
    # Overlap violations with soft-min
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    rs_sum = rs[i_idx] + rs[j_idx]
    
    # ov = max(0, r_i + r_j - dist)
    # Smooth: logsumexp(0, (r_i+r_j-dist)/beta) * beta
    ov = np.zeros(len(i_idx))
    for k in range(len(i_idx)):
        diff = rs_sum[k] - dist[k]
        ov[k] = np.log(np.exp(0) + np.exp(diff / beta)) * beta
    
    open_ = np.sum(ov ** 2)
    
    obj = -np.sum(rs) + 100.0 * (bpen + open_)
    
    # Gradient - this is approximate, use numerical if needed
    # For now, return objective only and use finite differences
    return obj

def _high_precision_polish(x, n=N, max_iter=500, tol=1e-12):
    """
    High-precision polish using exact gradient descent with adaptive step size.
    """
    x = x.copy()
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        energy, grad = _penalty_and_grad(x, n, mu=50000.0)
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Adaptive step size
        step = 0.01
        for _ in range(20):
            x_new = x - step * grad
            # Clip to bounds
            x_new[0::3] = np.clip(x_new[0::3], 0, 1)
            x_new[1::3] = np.clip(x_new[1::3], 0, 1)
            x_new[2::3] = np.clip(x_new[2::3], 0, 0.5)
            
            e_new, _ = _penalty_and_grad(x_new, n, mu=50000.0)
            if e_new < energy:
                x = x_new
                break
            step *= 0.5
        
        if step < 1e-15:
            break
    
    return x

def _is_valid(x, n=N, tol=1e-7):
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Check boundaries
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
        
    # Check overlaps
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True

def _fix_feasibility(x, n=N, tol=1e-8):
    """
    Aggressively shrink radii and adjust positions to ensure validity.
    This is a fallback to guarantee a valid output.
    """
    pts = x.reshape(n, 3).copy()
    for _ in range(200):
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # 1. Boundary fix: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
        max_r_bound = np.minimum(np.minimum(xs, ys), np.minimum(1.0 - xs, 1.0 - ys))
        max_r_bound = max_r_bound * (1 - 1e-6)
        rs = np.minimum(rs, max_r_bound)
        
        # 2. Overlap fix: If r_i + r_j > dist, shrink both
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)
        
        if not np.any(ov > tol):
            break
            
        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)
        
        reduction = overlap_per_circle * 0.5 + 1e-6
        rs = np.maximum(0.0, rs - reduction)
        
        pts[:, 2] = rs
        
    return pts.ravel()

def _local_search_refine(x, n=N, n_iter=50, step_size=0.01):
    """
    Local search with random perturbations to escape local minima.
    """
    x = x.copy()
    rng = np.random.default_rng(123)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    best_score = np.sum(x.reshape(n, 3)[:, 2])
    best_x = x.copy()
    
    for it in range(n_iter):
        if _remaining() < 3.0:
            break
        
        # Random perturbation
        x_trial = x.copy()
        for dim in range(3 * n):
            if rng.random() < 0.3:  # Perturb 30% of variables
                delta = rng.normal(0, step_size)
                x_trial[dim] += delta
                x_trial[dim] = np.clip(x_trial[dim], bounds[dim][0], bounds[dim][1])
        
        # Quick feasibility check and score
        x_valid = _fix_feasibility(x_trial)
        if _is_valid(x_valid):
            score = np.sum(x_valid.reshape(n, 3)[:, 2])
            if score > best_score:
                best_score = score
                best_x = x_valid.copy()
                x = x_valid.copy()
                # Reduce step size on improvement
                step_size *= 0.9
    
    return best_x

def construct():
    global _START
    _START = time.time()
    
    best_score = -np.inf
    best_x = None
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Generate diverse initializations
    inits = []
    
    # 1. Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    xs, ys, rs = [], [], []
    for i in range(side):
        for j in range(side):
            if len(xs) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            xs.append(x)
            ys.append(y)
            rs.append(r)
    while len(xs) < N:
        xs.append(0.5)
        ys.append(0.5)
        rs.append(0.05)
    inits.append(np.column_stack((xs, ys, rs)).ravel())
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(15):
        xs = rng.random(N)
        ys = rng.random(N)
        rs = rng.random(N) * 0.05 + 0.01
        inits.append(np.column_stack((xs, ys, rs)).ravel())
        
    # 3. Hexagonal-like initialization
    r_est = 0.095
    x_spacing = 2 * r_est
    y_spacing = r_est * np.sqrt(3)
    xs, ys, rs = [], [], []
    for row in range(7):
        count = 5 if row < 6 else 4
        if len(xs) + count > N:
            count = N - len(xs)
        y = (row + 0.5) * y_spacing
        for col in range(count):
            x = (col + 0.5) * x_spacing + (0.5 * x_spacing if row % 2 else 0)
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    if len(xs) < N:
        while len(xs) < N:
            xs.append(0.5)
            ys.append(0.5)
            rs.append(0.05)
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9) * 0.9 + 0.05
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9) * 0.9 + 0.05
    inits.append(np.column_stack((xs, ys, rs[:N])).ravel())
    
    # 4. Clustered initialization - groups of circles in corners and center
    for cluster_seed in [7, 13, 23]:
        rng_c = np.random.default_rng(cluster_seed)
        xs, ys, rs = [], [], []
        # Place some in corners, some in center
        n_corner = 8
        n_center = N - n_corner
        for i in range(n_corner):
            corner_x = rng_c.choice([0.1, 0.9])
            corner_y = rng_c.choice([0.1, 0.9])
            xs.append(corner_x + rng_c.normal(0, 0.05))
            ys.append(corner_y + rng_c.normal(0, 0.05))
            rs.append(0.08)
        for i in range(n_center):
            xs.append(0.5 + rng_c.normal(0, 0.15))
            ys.append(0.5 + rng_c.normal(0, 0.15))
            rs.append(0.07)
        xs = np.clip(xs, 0.05, 0.95)
        ys = np.clip(ys, 0.05, 0.95)
        inits.append(np.column_stack((xs, ys, rs)).ravel())
    
    # Optimize each initialization
    for i, x0 in enumerate(inits):
        if _remaining() < 8.0:
            break
            
        try:
            # Phase 1: L-BFGS-B with moderate penalty
            res = minimize(
                _penalty_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
            )
            x_opt = res.x
            
            # Phase 2: High-precision polish
            if _remaining() > 5.0:
                x_polished = _high_precision_polish(x_opt, max_iter=200, tol=1e-13)
            else:
                x_polished = x_opt
            
            # Ensure feasibility
            x_valid = _fix_feasibility(x_polished)
            
            # Phase 3: Local search refinement
            if _remaining() > 4.0:
                x_refined = _local_search_refine(x_valid, n_iter=30, step_size=0.008)
            else:
                x_refined = x_valid
            
            if _is_valid(x_refined):
                score = np.sum(x_refined.reshape(N, 3)[:, 2])
                if score > best_score:
                    best_score = score
                    best_x = x_refined.copy()
        except Exception:
            continue
            
    # Final polish on best solution
    if best_x is not None and _remaining() > 3.0:
        try:
            x_final = _high_precision_polish(best_x, max_iter=300, tol=1e-14)
            x_final = _fix_feasibility(x_final)
            if _is_valid(x_final):
                score = np.sum(x_final.reshape(N, 3)[:, 2])
                if score > best_score:
                    best_score = score
                    best_x = x_final.copy()
        except Exception:
            pass
    
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells)
        
    return best_x.reshape(N, 3)
