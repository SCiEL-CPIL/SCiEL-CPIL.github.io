import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 145.0  # Safety margin before 150s limit

def _objective(v):
    """
    Objective function: Minimize -sum(r).
    v: flattened array [x(26), y(26), r(26)]
    """
    r = v[2*N:3*N]
    return -r.sum()

def _objective_grad(v):
    """
    Gradient of the objective function.
    """
    grad = np.zeros_like(v)
    grad[2*N:3*N] = -1.0
    return grad

def _make_constraints():
    """
    Generate the constraint functions and their gradients for SLSQP.
    Constraints are of the form c(x) >= 0.
    """
    cons = []
    
    # Boundary constraints:
    # 1. x_i - r_i >= 0
    # 2. 1 - x_i - r_i >= 0
    # 3. y_i - r_i >= 0
    # 4. 1 - y_i - r_i >= 0
    
    # We can define these as a single function returning an array of values, 
    # but SLSQP expects a list of dicts or a single dict with 'fun' and 'jac'.
    # Let's create a single constraint function for boundaries to reduce overhead.
    
    def boundary_fun(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        c1 = x - r
        c2 = 1.0 - x - r
        c3 = y - r
        c4 = 1.0 - y - r
        return np.concatenate([c1, c2, c3, c4])
        
    def boundary_jac(v):
        # Shape (4*N, 3*N)
        # Rows correspond to c1[0..N-1], c2[0..N-1], c3[0..N-1], c4[0..N-1]
        # Cols correspond to x[0..N-1], y[0..N-1], r[0..N-1]
        J = np.zeros((4*N, 3*N))
        
        # c1 = x - r
        # dc1/dx = I, dc1/dr = -I
        J[0:N, 0:N] = np.eye(N)
        J[0:N, 2*N:3*N] = -np.eye(N)
        
        # c2 = 1 - x - r
        # dc2/dx = -I, dc2/dr = -I
        J[N:2*N, 0:N] = -np.eye(N)
        J[N:2*N, 2*N:3*N] = -np.eye(N)
        
        # c3 = y - r
        # dc3/dy = I, dc3/dr = -I
        J[2*N:3*N, N:2*N] = np.eye(N)
        J[2*N:3*N, 2*N:3*N] = -np.eye(N)
        
        # c4 = 1 - y - r
        # dc4/dy = -I, dc4/dr = -I
        J[3*N:4*N, N:2*N] = -np.eye(N)
        J[3*N:4*N, 2*N:3*N] = -np.eye(N)
        
        return J

    cons.append({'type': 'ineq', 'fun': boundary_fun, 'jac': boundary_jac})
    
    # Overlap constraints: dist(i,j) - r_i - r_j >= 0 for i < j
    # There are N*(N-1)/2 such constraints.
    # This might be too many for SLSQP to handle efficiently if we define them one by one.
    # Better to define a single function returning all overlap constraints.
    
    num_pairs = N * (N - 1) // 2
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def overlap_fun(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        vals = np.zeros(num_pairs)
        for k, (i, j) in enumerate(pairs):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            vals[k] = dist - r[i] - r[j]
        return vals
        
    def overlap_jac(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        J = np.zeros((num_pairs, 3*N))
        for k, (i, j) in enumerate(pairs):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx*dx + dy*dy
            if dist2 < 1e-12:
                dist = 1e-6
                # Gradient is undefined at 0, use small value to avoid division by zero
                # Actually, if dist is 0, circles overlap completely. 
                # We can just set gradient to 0 or small random? 
                # Better to handle this case in the objective or initialization.
                # For now, if dist is very small, we assume they are touching or overlapping.
                # The gradient of dist w.r.t x_i is (x_i-x_j)/dist.
                # If dist -> 0, this is undefined. 
                # We'll use a small epsilon.
                dx = 1e-6 if dx == 0 else dx
                dy = 1e-6 if dy == 0 else dy
                dist = np.sqrt(dx*dx + dy*dy)
            
            # d(dist)/dx_i = dx/dist
            J[k, i] = dx / dist
            # d(dist)/dx_j = -dx/dist
            J[k, j] = -dx / dist
            # d(dist)/dy_i = dy/dist
            J[k, N + i] = dy / dist
            # d(dist)/dy_j = -dy/dist
            J[k, N + j] = -dy / dist
            # d(-r_i)/dr_i = -1
            J[k, 2*N + i] = -1.0
            # d(-r_j)/dr_j = -1
            J[k, 2*N + j] = -1.0
            
        return J

    cons.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
    
    return cons

def _is_valid(arr, tol=1e-5):
    """Check if the arrangement is strictly valid."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Check radii non-negative
    if np.any(r < 0):
        return False
        
    # Check boundaries
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _make_grid_initial():
    """Grid initial guess: 5x5 grid plus 1 extra."""
    side = 6
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    pts = pts[:N]
    arr = np.array(pts)
    # Initial radius: half the cell size, slightly reduced
    r0 = (0.5 / side) * 0.9
    # Ensure it fits in boundaries
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])

def _make_hex_initial():
    """Hexagonal-ish initial guess."""
    pts = []
    # 5 rows, 5 cols
    for row in range(5):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.2
            pts.append([x, y])
    pts.append([0.5, 0.5]) # 26th point
    
    # Scale to fit [0,1] with margin
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    scale = 0.8 / max(xmax - xmin, ymax - ymin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    for p in pts:
        p[0] = 0.5 + (p[0] - cx) * scale
        p[1] = 0.5 + (p[1] - cy) * scale
    
    arr = np.array(pts)
    # Estimate radius from min distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((arr[i, 0] - arr[j, 0])**2 + (arr[i, 1] - arr[j, 1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.9
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])

def _make_random_initial(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    
    # Calculate min distance to neighbors for radius estimation
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    
    r = 0.4 * min_dists
    # Bound by walls
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.01)
    
    return np.column_stack([x, y, r])

def _optimize_single(init_arr, time_limit):
    """
    Run SLSQP optimization starting from init_arr.
    """
    v0 = init_arr.flatten()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    best_val = -np.inf
    best_v = v0.copy()
    
    # Check initial validity
    arr0 = v0.reshape(N, 3)
    if _is_valid(arr0):
        best_val = arr0[:, 2].sum()
        best_v = v0.copy()
    
    cons = _make_constraints()
    
    try:
        res = minimize(
            _objective, v0,
            jac=_objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-9}
        )
        v_res = res.x
        arr_res = v_res.reshape(N, 3)
        
        if _is_valid(arr_res):
            val = arr_res[:, 2].sum()
            if val > best_val:
                best_val = val
                best_v = v_res.copy()
                
    except Exception as e:
        # If SLSQP fails, try to recover or return best so far
        pass
        
    return best_v, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_arr = None
    best_val = -np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_make_grid_initial())
    inits.append(_make_hex_initial())
    
    # Random inits with different seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4000, 5000, 6000]
    for s in seeds:
        inits.append(_make_random_initial(s))
        
    # Try each initialization
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            v_best, val = _optimize_single(init, deadline - 5)
        except Exception:
            continue
            
        if val > best_val:
            best_val = val
            best_arr = v_best.reshape(N, 3)
            
    # If no valid solution found from optimization, use a repaired version of the best attempt
    if best_arr is None:
        # Fallback: use grid and repair
        best_arr = _make_grid_initial()
        best_val = best_arr[:, 2].sum()
        
    # Final Polish: If the best solution is slightly invalid, try to repair it
    # Or, if it is valid, try one more step
    if _is_valid(best_arr):
        # Try to improve further with a second pass if time permits
        if time.time() < deadline - 5:
            v0 = best_arr.flatten()
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            cons = _make_constraints()
            try:
                res = minimize(
                    _objective, v0,
                    jac=_objective_grad,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                arr_new = res.x.reshape(N, 3)
                if _is_valid(arr_new):
                    val_new = arr_new[:, 2].sum()
                    if val_new > best_val:
                        best_val = val_new
                        best_arr = arr_new
            except Exception:
                pass
    else:
        # Repair the invalid solution
        # Simple repair: shrink radii
        x, y, r = best_arr[:, 0].copy(), best_arr[:, 1].copy(), best_arr[:, 2].copy()
        for _ in range(100):
            max_viol = 0.0
            for i in range(N):
                d_wall = min(x[i], y[i], 1.0 - x[i], 1.0 - y[i])
                if r[i] > d_wall:
                    max_viol = max(max_viol, r[i] - d_wall)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if r[i] + r[j] > d:
                        max_viol = max(max_viol, r[i] + r[j] - d)
            if max_viol < 1e-8:
                break
            shrink = max_viol / 2.0
            r -= shrink
            r = np.maximum(r, 0.0)
        best_arr = np.column_stack([x, y, r])
        if _is_valid(best_arr):
            best_val = best_arr[:, 2].sum()
        else:
            # Last resort: return a known valid grid
            best_arr = _make_grid_initial()
            best_val = best_arr[:, 2].sum()
            
    return best_arr
