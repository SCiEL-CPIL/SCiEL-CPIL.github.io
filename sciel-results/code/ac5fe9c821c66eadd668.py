"""Rewritten program: constrained optimization for circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 21
DEADLINE = 115.0  # safety margin before 120s limit


def _validate(arr):
    """Check if array of circles is valid: disjoint interiors, bounding rect perimeter <= 4."""
    if arr.shape != (N, 3):
        return False
    if np.any(arr[:, 2] < 0):
        return False
    # Bounding rectangle
    xmin = np.min(arr[:, 0] - arr[:, 2])
    xmax = np.max(arr[:, 0] + arr[:, 2])
    ymin = np.min(arr[:, 1] - arr[:, 2])
    ymax = np.max(arr[:, 1] + arr[:, 2])
    W = xmax - xmin
    H = ymax - ymin
    if 2 * (W + H) > 4.0 + 1e-9:
        return False
    # Non-overlap: distance between centers >= sum of radii
    for i in range(N):
        for j in range(i + 1, N):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < arr[i, 2] + arr[j, 2] - 1e-9:
                return False
    return True


def _build_initial():
    """Create initial guess: grid of 21 circles in a rectangle with perimeter 4."""
    # Use a 5x5 grid, take 21 circles
    # Rectangle with perimeter 4: W + H = 2
    # Let's try W=1.4, H=0.6 (ratio ~ 2.33:1) or W=1.0, H=1.0
    # For 21 circles, a 5x5 grid is natural. Let's use W=1.5, H=0.5
    # Actually, let's use multiple aspect ratios and pick the best initial
    
    best_arr = None
    best_sum = 0
    
    for W in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
        H = 2.0 - W
        if H <= 0.1:
            continue
        # 5 columns, 5 rows, use 21
        cols = 5
        rows = 5
        # Centers
        xs = np.linspace(W / (2 * cols), W - W / (2 * cols), cols)
        ys = np.linspace(H / (2 * rows), H - H / (2 * rows), rows)
        # Radius: min distance to boundary and to neighbor / 2
        r = min(W / (2 * cols), H / (2 * rows)) * 0.95
        # Build 21 circles
        centers = []
        for i in range(cols):
            for j in range(rows):
                if len(centers) < N:
                    centers.append((xs[i], ys[j], r))
        arr = np.array(centers[:N])
        s = np.sum(arr[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr
    
    if best_arr is None:
        # Fallback: 5x5 grid in square
        W = H = 1.0
        cols = rows = 5
        xs = np.linspace(W / (2 * cols), W - W / (2 * cols), cols)
        ys = np.linspace(H / (2 * rows), H - H / (2 * rows), rows)
        r = min(W / (2 * cols), H / (2 * rows)) * 0.95
        centers = []
        for i in range(cols):
            for j in range(rows):
                if len(centers) < N:
                    centers.append((xs[i], ys[j], r))
        best_arr = np.array(centers[:N])
    
    return best_arr


def _objective_and_grad(x, pairs):
    """
    x: [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20] (63 variables)
    Maximize sum of radii (minimize negative sum)
    """
    n = len(pairs)
    # Extract
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Objective: minimize -sum(r)
    obj = -np.sum(rs)
    
    # Gradient
    grad = np.zeros_like(x)
    grad[2::3] = -1.0  # d(-sum r)/dr_i = -1
    
    # Constraint violations (for gradient of penalty)
    # We use a smooth penalty approach
    # Actually, let's use SLSQP with explicit constraints
    
    return obj, grad


def _make_constraints(pairs):
    """
    Create constraint functions for SLSQP.
    Constraints:
    1. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2 for all pairs
       => (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    2. Circles inside bounding box [0,W]x[0,H] where W+H=2
       But W and H are determined by the circles themselves...
       This is tricky. Let's use a fixed bounding box and then scale.
    
    Actually, let's reformulate: fix the bounding box to [0,2]x[0,0]... no.
    
    Better approach: Use a fixed rectangle [0, W] x [0, H] with W+H=2,
    and optimize over W as well, or fix W and H.
    
    Let's fix W and H, then optimize centers and radii.
    """
    pass


def _solve_for_fixed_rect(W, H, x0):
    """
    Solve for fixed rectangle [0,W]x[0,H] with W+H=2.
    Variables: centers (x_i, y_i) and radii r_i for i=0..20
    Constraints:
    - r_i >= 0
    - x_i >= r_i, x_i <= W - r_i
    - y_i >= r_i, y_i <= H - r_i
    - (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2 for all i<j
    """
    n = N
    # Variable vector: [x0, y0, r0, x1, y1, r1, ...]
    x = x0.copy()
    
    def obj(v):
        rs = v[2::3]
        return -np.sum(rs)
    
    def obj_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g
    
    constraints = []
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            ii, ij = i * 3, j * 3
            def non_overlap(v, ii=ii, ij=ij):
                xi, yi, ri = v[ii], v[ii+1], v[ii+2]
                xj, yj, rj = v[ij], v[ij+1], v[ij+2]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            def non_overlap_grad(v, ii=ii, ij=ij):
                xi, yi, ri = v[ii], v[ii+1], v[ii+2]
                xj, yj, rj = v[ij], v[ij+1], v[ij+2]
                g = np.zeros_like(v)
                g[ii] = 2 * (xi - xj)
                g[ii+1] = 2 * (yi - yj)
                g[ii+2] = -2 * (ri + rj)
                g[ij] = -2 * (xi - xj)
                g[ij+1] = -2 * (yi - yj)
                g[ij+2] = -2 * (ri + rj)
                return g
            constraints.append({'type': 'ineq', 'fun': non_overlap, 'jac': non_overlap_grad})
    
    # Boundary constraints
    for i in range(n):
        ii = i * 3
        # x_i >= r_i => x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii: v[ii] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([1.0 if k == ii else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
        # W - x_i >= r_i => W - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii, W=W: W - v[ii] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([-1.0 if k == ii else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
        # y_i >= r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii: v[ii+1] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([1.0 if k == ii+1 else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
        # H - y_i >= r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii, H=H: H - v[ii+1] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([-1.0 if k == ii+1 else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
    
    # Bounds
    bounds = []
    for i in range(n):
        ii = i * 3
        bounds.append((0.0, W))  # x_i
        bounds.append((0.0, H))  # y_i
        bounds.append((0.0, min(W, H) / 2))  # r_i
    
    result = minimize(
        obj, x, method='SLSQP', jac=obj_grad,
        bounds=bounds, constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    return result


def _try_aspect_ratios():
    """Try multiple aspect ratios and return the best result."""
    best_val = 0
    best_arr = None
    
    x0 = _build_initial()
    
    for W in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
        H = 2.0 - W
        if H < 0.2:
            continue
        
        # Scale initial guess to fit this rectangle
        # Get current bounding box of x0
        xmin = np.min(x0[:, 0] - x0[:, 2])
        xmax = np.max(x0[:, 0] + x0[:, 2])
        ymin = np.min(x0[:, 1] - x0[:, 2])
        ymax = np.max(x0[:, 1] + x0[:, 2])
        cur_W = xmax - xmin
        cur_H = ymax - ymin
        
        if cur_W <= 0 or cur_H <= 0:
            continue
        
        scale_x = W / cur_W
        scale_y = H / cur_H
        scale = min(scale_x, scale_y) * 0.9  # margin
        
        x0_scaled = x0.copy()
        x0_scaled[:, 0] = (x0_scaled[:, 0] - xmin) * scale + xmin * 0  # shift to start at 0
        x0_scaled[:, 1] = (x0_scaled[:, 1] - ymin) * scale
        x0_scaled[:, 2] = x0_scaled[:, 2] * scale
        
        # Ensure within bounds
        x0_scaled[:, 0] = np.clip(x0_scaled[:, 0], x0_scaled[:, 2], W - x0_scaled[:, 2])
        x0_scaled[:, 1] = np.clip(x0_scaled[:, 1], x0_scaled[:, 2], H - x0_scaled[:, 2])
        
        result = _solve_for_fixed_rect(W, H, x0_scaled)
        
        if result.success or result.fun is not None:
            arr = result.x.reshape(N, 3)
            if _validate(arr):
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best_arr = arr
    
    return best_arr, best_val


def _refine(arr, W, H):
    """Refine a valid arrangement using local optimization."""
    x = arr.flatten()
    
    def obj(v):
        return -np.sum(v[2::3])
    
    def obj_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g
    
    constraints = []
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            ii, ij = i * 3, j * 3
            def non_overlap(v, ii=ii, ij=ij):
                xi, yi, ri = v[ii], v[ii+1], v[ii+2]
                xj, yj, rj = v[ij], v[ij+1], v[ij+2]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            def non_overlap_grad(v, ii=ii, ij=ij):
                xi, yi, ri = v[ii], v[ii+1], v[ii+2]
                xj, yj, rj = v[ij], v[ij+1], v[ij+2]
                g = np.zeros_like(v)
                g[ii] = 2 * (xi - xj)
                g[ii+1] = 2 * (yi - yj)
                g[ii+2] = -2 * (ri + rj)
                g[ij] = -2 * (xi - xj)
                g[ij+1] = -2 * (yi - yj)
                g[ij+2] = -2 * (ri + rj)
                return g
            constraints.append({'type': 'ineq', 'fun': non_overlap, 'jac': non_overlap_grad})
    
    for i in range(n):
        ii = i * 3
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii: v[ii] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([1.0 if k == ii else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii, W=W: W - v[ii] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([-1.0 if k == ii else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii: v[ii+1] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([1.0 if k == ii+1 else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, ii=ii, H=H: H - v[ii+1] - v[ii+2],
            'jac': lambda v, ii=ii: np.array([-1.0 if k == ii+1 else (-1.0 if k == ii+2 else 0.0) for k in range(len(v))])
        })
    
    bounds = []
    for i in range(n):
        ii = i * 3
        bounds.append((0.0, W))
        bounds.append((0.0, H))
        bounds.append((0.0, min(W, H) / 2))
    
    result = minimize(
        obj, x, method='SLSQP', jac=obj_grad,
        bounds=bounds, constraints=constraints,
        options={'maxiter': 300, 'ftol': 1e-10}
    )
    
    if result.fun is not None:
        arr_new = result.x.reshape(N, 3)
        if _validate(arr_new):
            return arr_new
    return arr


def construct():
    start_time = time.time()
    
    # Get initial solution
    x0 = _build_initial()
    base_val = np.sum(x0[:, 2])
    best_arr = x0.copy()
    best_val = base_val
    
    # Try multiple aspect ratios
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_arr
    
    for W in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        H = 2.0 - W
        if H < 0.2:
            continue
        
        # Scale initial guess
        xmin = np.min(x0[:, 0] - x0[:, 2])
        xmax = np.max(x0[:, 0] + x0[:, 2])
        ymin = np.min(x0[:, 1] - x0[:, 2])
        ymax = np.max(x0[:, 1] + x0[:, 2])
        cur_W = xmax - xmin
        cur_H = ymax - ymin
        
        if cur_W <= 0 or cur_H <= 0:
            continue
        
        scale_x = W / cur_W
        scale_y = H / cur_H
        scale = min(scale_x, scale_y) * 0.85
        
        x0_scaled = x0.copy()
        x0_scaled[:, 0] = (x0_scaled[:, 0] - xmin) * scale
        x0_scaled[:, 1] = (x0_scaled[:, 1] - ymin) * scale
        x0_scaled[:, 2] = x0_scaled[:, 2] * scale
        
        x0_scaled[:, 0] = np.clip(x0_scaled[:, 0], x0_scaled[:, 2], W - x0_scaled[:, 2])
        x0_scaled[:, 1] = np.clip(x0_scaled[:, 1], x0_scaled[:, 2], H - x0_scaled[:, 2])
        
        result = _solve_for_fixed_rect(W, H, x0_scaled)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        if result.x is not None:
            arr = result.x.reshape(N, 3)
            if _validate(arr):
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best_arr = arr.copy()
    
    # Final refinement on best arrangement
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Determine W, H from best_arr
        xmin = np.min(best_arr[:, 0] - best_arr[:, 2])
        xmax = np.max(best_arr[:, 0] + best_arr[:, 2])
        ymin = np.min(best_arr[:, 1] - best_arr[:, 2])
        ymax = np.max(best_arr[:, 1] + best_arr[:, 2])
        W = xmax - xmin
        H = ymax - ymin
        if W + H > 2.0:
            W = 1.0
            H = 1.0
        refined = _refine(best_arr, W, H)
        if np.sum(refined[:, 2]) > best_val:
            best_arr = refined
            best_val = np.sum(best_arr[:, 2])
    
    # Final deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return best valid
        pass
    
    return best_arr
