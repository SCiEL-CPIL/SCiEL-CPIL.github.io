"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize.constraints import NonlinearConstraint

def _validate(x, y, r, w, h, tol=1e-5):
    """
    Validate if the configuration is feasible.
    Returns True if all constraints are satisfied within tolerance.
    """
    if w + h > 2.0 + 1e-4:
        return False
    if w < 1e-8 or h < 1e-8:
        return False
    
    # Check radii positivity
    if np.any(r < 1e-8):
        return False
        
    # Check boundary constraints
    # x_i - r_i >= 0, x_i + r_i <= w
    # y_i - r_i >= 0, y_i + r_i <= h
    if np.any(x - r < -tol) or np.any(x + r > w + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > h + tol):
        return False
        
    # Check non-overlap constraints
    # dist(i,j) >= r_i + r_j
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum * rad_sum - tol:
                return False
    return True

def _objective(v, w, h):
    """
    Objective: minimize -sum(r)
    v = [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
    """
    r = v[2*N:3*N]
    return -np.sum(r)

def _objective_grad(v, w, h):
    """
    Gradient of objective.
    d/dx = 0, d/dy = 0, d/dr = -1
    """
    grad = np.zeros(3*N)
    grad[2*N:3*N] = -1.0
    return grad

def _boundary_constraints(v, w, h):
    """
    Returns array of boundary constraint values.
    Constraints:
    1. x_i - r_i >= 0
    2. w - (x_i + r_i) >= 0
    3. y_i - r_i >= 0
    4. h - (y_i + r_i) >= 0
    
    Total 4*N constraints.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    c1 = x - r
    c2 = w - (x + r)
    c3 = y - r
    c4 = h - (y + r)
    
    return np.concatenate([c1, c2, c3, c4])

def _boundary_jacobian(v, w, h):
    """
    Jacobian of boundary constraints.
    Rows correspond to c1, c2, c3, c4 for each i.
    """
    jac = np.zeros((4*N, 3*N))
    for i in range(N):
        # c1_i = x_i - r_i
        jac[0*N + i, i] = 1.0
        jac[0*N + i, 2*N + i] = -1.0
        
        # c2_i = w - x_i - r_i
        jac[1*N + i, i] = -1.0
        jac[1*N + i, 2*N + i] = -1.0
        
        # c3_i = y_i - r_i
        jac[2*N + i, N + i] = 1.0
        jac[2*N + i, 2*N + i] = -1.0
        
        # c4_i = h - y_i - r_i
        jac[3*N + i, N + i] = -1.0
        jac[3*N + i, 2*N + i] = -1.0
    return jac

def _overlap_constraints(v, w, h):
    """
    Returns array of non-overlap constraint values.
    Constraint: dist(i,j)^2 - (r_i + r_j)^2 >= 0
    This is quadratic in v.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            rad_sum = r[i] + r[j]
            val = dist_sq - rad_sum*rad_sum
            cons.append(val)
    return np.array(cons)

def _overlap_jacobian(v, w, h):
    """
    Jacobian of non-overlap constraints.
    f_ij = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2
    df/dx_i = 2(x_i-x_j), df/dx_j = -2(x_i-x_j)
    df/dy_i = 2(y_i-y_j), df/dy_j = -2(y_i-y_j)
    df/dr_i = -2(r_i+r_j), df/dr_j = -2(r_i+r_j)
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    num_pairs = N*(N-1)//2
    jac = np.zeros((num_pairs, 3*N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            rad_sum = r[i] + r[j]
            
            jac[idx, i] = 2 * dx
            jac[idx, j] = -2 * dx
            jac[idx, N + i] = 2 * dy
            jac[idx, N + j] = -2 * dy
            jac[idx, 2*N + i] = -2 * rad_sum
            jac[idx, 2*N + j] = -2 * rad_sum
            idx += 1
    return jac

def _solve_aspect(w, h, deadline):
    """
    Solve the QCP for a fixed aspect ratio w x h.
    Returns best_val, best_circles or None if failed.
    """
    # Initial guess: Grid
    # Arrange in a grid to minimize initial overlaps
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    
    for i in range(N):
        c = i % cols
        r = i // cols
        x0[i] = (c + 0.5) * w / cols
        y0[i] = (r + 0.5) * h / rows
        r0[i] = 0.4 * min(w/cols, h/rows)
        
    v0 = np.concatenate([x0, y0, r0])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
    for i in range(N):
        bounds.append((0, h))  # y
    for i in range(N):
        bounds.append((1e-6, min(w, h)/2))  # r
        
    # Constraints
    # Boundary
    cons_boundary = NonlinearConstraint(
        _boundary_constraints, 0, np.inf,
        jac=_boundary_jacobian,
        args=(w, h)
    )
    # Overlap
    cons_overlap = NonlinearConstraint(
        _overlap_constraints, 0, np.inf,
        jac=_overlap_jacobian,
        args=(w, h)
    )
    
    constraints = [cons_boundary, cons_overlap]
    
    # Options
    options = {
        'maxiter': 500,
        'gtol': 1e-8,
        'xtol': 1e-10,
        'verbose': 0
    }
    
    try:
        res = minimize(
            _objective, v0,
            method='trust-constr',
            jac=_objective_grad,
            hess=None, # Let it approximate
            hessp=None,
            bounds=bounds,
            constraints=constraints,
            args=(w, h),
            options=options
        )
    except Exception as e:
        return None, None
        
    if res.success or (res.status == 1): # 1 means max iterations reached but might be good
        v_opt = res.x
        x_opt = v_opt[:N]
        y_opt = v_opt[N:2*N]
        r_opt = v_opt[2*N:3*N]
        
        if _validate(x_opt, y_opt, r_opt, w, h):
            val = np.sum(r_opt)
            circles = np.column_stack([x_opt, y_opt, r_opt])
            return val, circles
            
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_val = 0.0
    best_circles = None
    
    # Fallback: Simple grid in 1x1 square
    # Ensure valid fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.45 / side # Small enough to not overlap in grid
            cells.append((cx, cy, cr))
    fallback = np.array(cells[:N])
    if _validate(fallback[:, 0], fallback[:, 1], fallback[:, 2], 1.0, 1.0):
        best_val = np.sum(fallback[:, 2])
        best_circles = fallback.copy()
    else:
        # Even smaller radii
        cr = 0.4 / side
        fallback[:, 2] = cr
        if _validate(fallback[:, 0], fallback[:, 1], fallback[:, 2], 1.0, 1.0):
            best_val = np.sum(fallback[:, 2])
            best_circles = fallback.copy()
            
    # Try various aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    # We explore w from 0.5 to 1.5
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 1.3, 0.9, 1.1, 1.4, 0.75, 1.25, 0.95]
    
    # Shuffle to ensure diversity if time runs out? No, keep deterministic order for reproducibility
    # But we can prioritize likely good ones. Square is usually good for packing.
    
    for w in aspect_ratios:
        if time.time() > deadline - 10: # Leave 10s buffer
            break
            
        h = 2.0 - w
        if h <= 0:
            continue
            
        val, circles = _solve_aspect(w, h, deadline)
        
        if val is not None and val > best_val:
            best_val = val
            best_circles = circles
            
    if best_circles is None:
        return fallback
        
    return best_circles
# EVOLVE-BLOCK-END
