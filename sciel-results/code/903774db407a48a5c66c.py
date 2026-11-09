"""Constrained SQP-based circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
T_LIMIT = 115.0  # safety margin under 120s
T_START = time.time()


def _elapsed():
    return time.time() - T_START


def _is_valid(x, W, H, tol=1e-5):
    n = len(x) // 3
    pos = x.reshape(n, 3)
    px, py, pr = pos[:, 0], pos[:, 1], pos[:, 2]
    if np.any(pr < -tol):
        return False
    if np.any(px < -tol) or np.any(px > W + tol) or np.any(py < -tol) or np.any(py > H + tol):
        return False
    # wall
    if np.any(pr - px > tol) or np.any(pr - (W - px) > tol) or \
       np.any(pr - py > tol) or np.any(pr - (H - py) > tol):
        return False
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    req = pr[:, None] + pr[None, :]
    if np.any(req - dist > tol):
        return False
    return True


def _init_hex(W, H, rng):
    """Initialize with a hexagonal lattice pattern scaled to W, H."""
    n = N
    # Estimate grid size
    cols = int(np.ceil(np.sqrt(N * W / H))) if W >= H else int(np.ceil(np.sqrt(N * H / W)))
    cols = max(1, min(cols, N))
    rows = int(np.ceil(N / cols))
    
    px = np.empty(N)
    py = np.empty(N)
    pr = np.empty(N)
    
    # Hexagonal spacing
    # We want to fit 'rows' rows and 'cols' columns.
    # Spacing in x: W / cols
    # Spacing in y: H / rows
    # For hex, vertical spacing is sqrt(3)/2 * horizontal spacing? 
    # Let's just use a distorted hex grid.
    
    dx = W / cols
    dy = H / rows
    
    idx = 0
    for j in range(rows):
        for i in range(cols):
            if idx >= N:
                break
            # Hex offset for odd rows
            offset = dx / 2.0 if j % 2 != 0 else 0.0
            # Keep inside bounds
            cx = (i + 0.5) * dx + offset
            cy = (j + 0.5) * dy
            
            # Clamp to rectangle
            cx = np.clip(cx, 0.0, W)
            cy = np.clip(cy, 0.0, H)
            
            # Add jitter
            cx += rng.uniform(-0.02, 0.02) * dx
            cy += rng.uniform(-0.02, 0.02) * dy
            
            # Radius estimate: half the minimum spacing
            r_est = 0.45 * min(dx, dy)
            
            px[idx] = cx
            py[idx] = cy
            pr[idx] = r_est
            idx += 1
            
    x = np.empty(3 * N)
    x[0::3] = px
    x[1::3] = py
    x[2::3] = pr
    return x


def _solve_constrained(x0, rng):
    """
    Solve for x = [x0, y0, r0, ..., x20, y20, r20, W, H]
    Objective: minimize -sum(r_i)
    Constraints:
    1. W + H = 2 (Equality)
    2. W > 0.1, H > 0.1 (Inequality via bounds)
    3. Circle i inside [0, W] x [0, H] (Inequalities)
    4. Circle i non-overlap Circle j (Inequalities)
    """
    n = N
    # Variables: 3N + 2
    # Bounds
    lb = np.zeros(3 * n + 2)
    ub = np.zeros(3 * n + 2)
    
    # W, H bounds (indices 3N, 3N+1)
    lb[3 * n] = 0.1
    ub[3 * n] = 1.9
    lb[3 * n + 1] = 0.1
    ub[3 * n + 1] = 1.9
    
    # Circle bounds (dynamic, but we can set static ones for x,y,r relative to max possible W,H)
    # Since W+H=2, max W=1.9, max H=1.9.
    # x in [0, 1.9], y in [0, 1.9], r in [0, 0.95]
    for i in range(n):
        lb[3 * i] = 0.0
        ub[3 * i] = 1.9
        lb[3 * i + 1] = 0.0
        ub[3 * i + 1] = 1.9
        lb[3 * i + 2] = 1e-4
        ub[3 * i + 2] = 0.95
        
    # Objective function
    def objective(v):
        # v[0:3N] are circles, v[3N] is W, v[3N+1] is H
        r = v[2::3][:n]
        return -np.sum(r)
        
    # Constraints
    # Equality: W + H = 2
    def eq_constraint(v):
        return v[3 * n] + v[3 * n + 1] - 2.0
        
    # Inequalities:
    # 1. Walls:
    #    x_i - r_i >= 0
    #    W - x_i - r_i >= 0
    #    y_i - r_i >= 0
    #    H - y_i - r_i >= 0
    # 2. Non-overlap:
    #    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def ineq_constraints(v):
        pos = v[:3 * n].reshape(n, 3)
        px, py, pr = pos[:, 0], pos[:, 1], pos[:, 2]
        W = v[3 * n]
        H = v[3 * n + 1]
        
        cons = []
        # Walls
        cons.append(px - pr)
        cons.append(W - px - pr)
        cons.append(py - pr)
        cons.append(H - py - pr)
        
        # Non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                dist_sq = dx * dx + dy * dy
                req = pr[i] + pr[j]
                cons.append(dist_sq - req * req)
                
        return np.concatenate(cons)
        
    cons = [{'type': 'eq', 'fun': eq_constraint}]
    # We can't easily pass a single function for all inequalities to SLSQP if we want to be efficient,
    # but SLSQP accepts a list of dicts. 
    # Actually, SLSQP accepts a list of dicts for constraints.
    # Let's define a single inequality function that returns an array.
    # scipy.optimize.minimize with method='SLSQP' accepts 'ineq' constraints as a list of dicts, 
    # where each dict has 'fun' returning a scalar or array? 
    # According to docs: "ineq" constraint functions must return an array of length 1 (scalar) if used in a list? 
    # No, if you pass a list, each element is a constraint. 
    # However, for efficiency, it is better to define one function that returns all inequalities if the solver supports it.
    # SLSQP does support vector-valued inequality constraints if passed as a single dict with 'fun' returning an array?
    # Let's check. "The constraint function should return a scalar or an array." 
    # If I return an array, it treats each element as a separate constraint >= 0.
    
    def ineq_func(v):
        return ineq_constraints(v)
        
    cons.append({'type': 'ineq', 'fun': ineq_func})
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=list(zip(lb, ub)), 
                       constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
        return res.x
    except Exception as e:
        # Fallback to penalty if SLSQP fails or is too slow
        return None


def construct():
    global T_START
    T_START = time.time()
    rng = np.random.default_rng(20240501)

    best_x = None
    best_score = -np.inf
    best_W = 1.0
    best_H = 1.0

    # Fallback seed
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    base = np.array(cells[:N])
    # Ensure base is valid in 1x1
    if _is_valid(base.ravel(), 1.0, 1.0, tol=1e-4):
        best_score = np.sum(base[:, 2])
        # Store in format [x,y,r,...,W,H]
        base_full = np.zeros(3 * N + 2)
        base_full[:3 * N] = base.ravel()
        base_full[3 * N] = 1.0
        base_full[3 * N + 1] = 1.0
        best_x = base_full
        best_W = 1.0
        best_H = 1.0

    # Multistart
    restarts = 0
    max_restarts = 40
    while restarts < max_restarts and _elapsed() < T_LIMIT - 10:
        restarts += 1
        
        # Random aspect ratio
        ar = rng.uniform(0.5, 2.0)
        H = 2.0 / (1.0 + ar)
        W = 2.0 - H
        
        # Initialize
        x0_circles = _init_hex(W, H, rng)
        x0 = np.zeros(3 * N + 2)
        x0[:3 * N] = x0_circles
        x0[3 * N] = W
        x0[3 * N + 1] = H
        
        # Solve
        res_x = _solve_constrained(x0, rng)
        
        if res_x is not None:
            # Extract W, H
            W_res = res_x[3 * N]
            H_res = res_x[3 * N + 1]
            circles = res_x[:3 * N].reshape(N, 3)
            
            # Validate
            if _is_valid(circles.ravel(), W_res, H_res, tol=1e-4):
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_x = res_x.copy()
                    best_W = W_res
                    best_H = H_res

    if best_x is None:
        # Return fallback
        return base

    # Return only the circles (n, 3)
    out = best_x[:3 * N].reshape(N, 3)
    return out
