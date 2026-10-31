"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth differentiable objective with soft collision penalties,
optimized via L-BFGS-B from multiple diverse starts, then SLSQP polish."""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-7):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-7):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1 + exp(beta*x))/beta."""
    # For stability: if beta*x > 50, softplus ≈ x
    # if beta*x < -50, softplus ≈ 0
    bx = beta * x
    mask_large = bx > 50
    mask_small = bx < -50
    result = np.zeros_like(bx, dtype=float)
    mid = ~mask_large & ~mask_small
    result[mid] = np.log1p(np.exp(bx[mid])) / beta
    result[mask_large] = x[mask_large]
    # small: ~0
    return result


def _soft_max(x, beta=100.0):
    """Soft maximum of a set of values along last axis."""
    # max ≈ (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, subtract max
    max_x = np.max(x, axis=-1, keepdims=True)
    shifted = beta * (x - max_x)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return max_x + np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True)) / beta


def _objective(params, beta=100.0):
    """
    Objective: maximize sum of radii, penalize overlaps and boundary violations.
    Returns negative objective for minimization.
    
    params: array of shape (3*N,) = [x0, y0, r0, x1, y1, r1, ...]
    """
    n = N
    # Extract positions and radii
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Sum of radii (to maximize)
    sum_r = np.sum(rs)
    
    # Boundary penalties: want x >= r, x <= 1-r, y >= r, y <= 1-r
    # Penalty for x < r: softplus(r - x)
    # Penalty for x > 1-r: softplus(x - (1-r))
    # Similarly for y
    pen_x_left = np.sum(_softplus(rs - xs, beta))
    pen_x_right = np.sum(_softplus(xs - (1.0 - rs), beta))
    pen_y_left = np.sum(_softplus(rs - ys, beta))
    pen_y_right = np.sum(_softplus(ys - (1.0 - rs), beta))
    
    # Overlap penalties: for each pair (i,j), want dist >= ri + rj
    # Penalty: softplus(ri + rj - dist)
    # We compute all pairwise distances
    # For efficiency, we can vectorize
    # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # We want to penalize when ri + rj - dist > 0
    
    # Build pairwise matrices
    # To save memory, do it in chunks or use a loop for 26 circles (325 pairs)
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            gap = rs[i] + rs[j] - dist
            overlap_pen += _softplus(gap, beta)
    
    # Total penalty
    total_pen = pen_x_left + pen_x_right + pen_y_left + pen_y_right + overlap_pen
    
    # Objective: minimize -sum_r + lambda * total_pen
    # We want the penalty to dominate when violations exist
    # Use a large multiplier
    lam = 50.0
    return -sum_r + lam * total_pen


def _objective_grad(params, beta=100.0):
    """
    Gradient of the objective.
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    grad = np.zeros(3 * n)
    
    # Gradient of -sum_r: -1 for each radius
    grad[2::3] = -1.0
    
    # Boundary penalties
    # d/dx softplus(r - x) = -sigmoid(beta*(r-x))
    # d/dx softplus(x - (1-r)) = sigmoid(beta*(x - (1-r)))
    # d/dr softplus(r - x) = sigmoid(beta*(r-x))
    # d/dr softplus(x - (1-r)) = -sigmoid(beta*(x - (1-r)))
    
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    
    # x-left: softplus(r - x)
    z_xl = beta * (rs - xs)
    sig_xl = sigmoid(z_xl)
    grad[0::3] += -sig_xl  # d/dx
    grad[2::3] += sig_xl   # d/dr
    
    # x-right: softplus(x - (1-r))
    z_xr = beta * (xs - (1.0 - rs))
    sig_xr = sigmoid(z_xr)
    grad[0::3] += sig_xr   # d/dx
    grad[2::3] += -sig_xr  # d/dr
    
    # y-left: softplus(r - y)
    z_yl = beta * (rs - ys)
    sig_yl = sigmoid(z_yl)
    grad[1::3] += -sig_yl  # d/dy
    grad[2::3] += sig_yl   # d/dr
    
    # y-right: softplus(y - (1-r))
    z_yr = beta * (ys - (1.0 - rs))
    sig_yr = sigmoid(z_yr)
    grad[1::3] += sig_yr   # d/dy
    grad[2::3] += -sig_yr  # d/dr
    
    # Overlap penalties
    lam = 50.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            gap = rs[i] + rs[j] - dist
            z = beta * gap
            sig = sigmoid(z)
            
            # d/dgap softplus(gap) = sigmoid(beta*gap)
            # d/dx_i dist = dx / dist
            # d/dx_j dist = -dx / dist
            # d/dr_i gap = 1 - 0 (dist doesn't depend on r) -> actually d/dr_i (ri+rj-dist) = 1
            # Wait: gap = ri + rj - dist. dist depends on x,y only.
            # d/dr_i gap = 1
            # d/dr_j gap = 1
            # d/dx_i gap = -d/dx_i dist = -dx/dist
            # d/dx_j gap = -d/dx_j dist = +dx/dist
            # d/dy_i gap = -dy/dist
            # d/dy_j gap = +dy/dist
            
            # Gradient contribution to objective: lam * sig * d(gap)/d(param)
            gx_i = -dx / dist
            gx_j = dx / dist
            gy_i = -dy / dist
            gy_j = dy / dist
            gr_i = 1.0
            gr_j = 1.0
            
            grad[3*i] += lam * sig * gx_i
            grad[3*i+1] += lam * sig * gy_i
            grad[3*i+2] += lam * sig * gr_i
            grad[3*j] += lam * sig * gx_j
            grad[3*j+1] += lam * sig * gy_j
            grad[3*j+2] += lam * sig * gr_j
    
    return grad


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    rows_counts = [5, 5, 5, 5, 6]
    circles = []
    r = 0.08
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows_counts:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.02
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.45 / side
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.02, 0.08)
        circles.append((x, y, r))
    return circles


def _init_corner_biased(seed=None):
    """Initialization with circles biased toward corners."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    circles = []
    # Place some circles near corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        circles.append((cx + random.uniform(-0.05, 0.05), cy + random.uniform(-0.05, 0.05), 0.08))
    # Fill the rest randomly
    while len(circles) < N:
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.07)
        circles.append((x, y, r))
    return circles


def _lbgfbs_b_optimize(circles, deadline, beta_start=50.0, beta_end=200.0, max_iter=300):
    """
    Optimize using L-BFGS-B with annealing beta.
    Returns optimized circles.
    """
    params = np.array([val for circle in circles for val in circle])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    best_params = params.copy()
    best_obj = _objective(params, beta_start)
    
    # Anneal beta from beta_start to beta_end
    n_anneal_steps = 5
    for step in range(n_anneal_steps):
        if time.time() > deadline - 10.0:
            break
        beta = beta_start + (beta_end - beta_start) * (step / max(1, n_anneal_steps - 1))
        
        def obj_func(p):
            return _objective(p, beta)
        
        def grad_func(p):
            return _objective_grad(p, beta)
        
        try:
            result = minimize(
                obj_func, params, jac=grad_func,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter // n_anneal_steps, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
            params = result.x
        except Exception:
            pass
    
    # Convert back to circles
    circles_out = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(N)]
    return circles_out


def _slsqp_polish(circles, deadline):
    """SLSQP polish with pairwise constraints."""
    def params_to_circles(params):
        return [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    
    def objective(params):
        return -sum(params[3*i+2] for i in range(N))
    
    def _jac_overlap(p, i, j):
        jac = np.zeros(3 * N)
        xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
        xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
        jac[3*i] = 2 * (xi - xj)
        jac[3*j] = -2 * (xi - xj)
        jac[3*i+1] = 2 * (yi - yj)
        jac[3*j+1] = -2 * (yi - yj)
        jac[3*i+2] = -2 * (ri + rj)
        jac[3*j+2] = -2 * (ri + rj)
        return jac
    
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2], 'jac': lambda p, i=i: np.array([1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2], 'jac': lambda p, i=i: np.array([(-1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: np.array([1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: np.array([(-1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2], 'jac': lambda p, i=i: np.array([1.0 if k==3*i+2 else 0.0 for k in range(3*N)])})
    
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2,
                'jac': lambda p, i=i, j=j: _jac_overlap(p, i, j)
            })
    
    params = np.array([val for circle in circles for val in circle])
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    try:
        result = minimize(
            objective, params, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        if result.success:
            new_circles = params_to_circles(result.x)
            if _is_valid(new_circles):
                return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Multiple diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_corner_biased(999),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    for init_circles in inits:
        if time.time() > deadline - 20.0:
            break
        
        # L-BFGS-B optimization with annealing
        optimized = _lbgfbs_b_optimize(init_circles, deadline)
        
        # Check validity and sum
        if _is_valid(optimized):
            opt_sum = _sum_radii(optimized)
            if opt_sum > best_sum:
                best_circles = optimized
                best_sum = opt_sum
    
    # SLSQP polish on best
    if time.time() < deadline - 30.0:
        polished = _slsqp_polish(best_circles, deadline)
        polished_sum = _sum_radii(polished)
        if polished_sum > best_sum and _is_valid(polished):
            best_circles = polished
            best_sum = polished_sum
    
    # Final post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        # Fallback to a simple valid configuration
        best_circles = _init_grid()
        best_circles = _post_process(best_circles)
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
