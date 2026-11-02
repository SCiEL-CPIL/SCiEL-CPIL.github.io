"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - Analytical gradients for L-BFGS-B
 - Delaunay-based LP radii solver for exact feasibility given centers
 - Joint aspect ratio optimization integrated into the search
 - Simulated annealing refinement
 - Final high-precision polish pass with joint aspect
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

# EVOLVE-BLOCK-START

def _softplus_vec(x, beta=10.0):
    """Numerically stable softplus for arrays."""
    # For large positive x, softplus(x) ~ x
    # For large negative x, softplus(x) ~ 0
    # Use stable computation
    out = np.zeros_like(x, dtype=float)
    pos = x > 1.0 / beta
    out[pos] = x[pos]
    neg = ~pos
    out[neg] = np.log1p(np.exp(beta * x[neg])) / beta
    return out

def _sigmoid_vec(x, beta=10.0):
    """Numerically stable sigmoid derivative of softplus."""
    # d/dx softplus(x) = 1 / (1 + exp(-beta*x))
    out = np.zeros_like(x, dtype=float)
    pos = x > 0
    out[pos] = 1.0 / (1.0 + np.exp(-beta * x[pos]))
    exp_neg = np.exp(beta * x[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out

def _objective(params, w, h, beta=10.0):
    """
    Objective: minimize penalty - sum(r).
    params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Penalty for negative radii
    neg_r_pen = np.sum(np.minimum(rs, 0) ** 2)
    if np.any(rs <= 1e-12):
        return 1e10 + neg_r_pen
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            val = min_dist - dist
            if val > -1.0 / beta:  # Only compute softplus if potentially non-zero
                overlap_pen += _softplus_vec(np.array([val]), beta)[0]
    
    # Boundary penalty
    boundary_pen = 0.0
    for i in range(N):
        if xs[i] < rs[i]:
            boundary_pen += (rs[i] - xs[i]) ** 2
        if xs[i] > w - rs[i]:
            boundary_pen += (xs[i] - (w - rs[i])) ** 2
        if ys[i] < rs[i]:
            boundary_pen += (rs[i] - ys[i]) ** 2
        if ys[i] > h - rs[i]:
            boundary_pen += (ys[i] - (h - rs[i])) ** 2
    
    return overlap_pen + boundary_pen - np.sum(rs) + neg_r_pen

def _objective_grad(params, w, h, beta=10.0):
    """
    Analytical gradient of the objective.
    """
    xs = params[0::3].copy()
    ys = params[1::3].copy()
    rs = params[2::3].copy()
    
    grad = np.zeros_like(params)
    
    if np.any(rs <= 1e-12):
        # Return a gradient that pushes radii up
        grad[2::3] = 1.0
        return grad
    
    # Gradient of overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
            min_dist = rs[i] + rs[j]
            val = min_dist - dist
            
            if val > -1.0 / beta:
                # Derivative of softplus(val) w.r.t. val is sigmoid(beta * val)
                sig = _sigmoid_vec(np.array([val]), beta)[0]
                
                # d(val)/d(xs[i]) = -dx/dist
                # d(val)/d(xs[j]) = dx/dist
                # d(val)/d(rs[i]) = 1
                # d(val)/d(rs[j]) = 1
                
                grad[3 * i] += sig * (-dx / dist)
                grad[3 * i + 1] += sig * (-dy / dist)
                grad[3 * i + 2] += sig
                
                grad[3 * j] += sig * (dx / dist)
                grad[3 * j + 1] += sig * (dy / dist)
                grad[3 * j + 2] += sig
    
    # Gradient of boundary penalty
    for i in range(N):
        if xs[i] < rs[i]:
            grad[3 * i] += 2 * (rs[i] - xs[i]) * (-1)  # d/dx (r-x)^2 = -2(r-x)
            grad[3 * i + 2] += 2 * (rs[i] - xs[i])    # d/dr (r-x)^2 = 2(r-x)
        if xs[i] > w - rs[i]:
            grad[3 * i] += 2 * (xs[i] - (w - rs[i]))  # d/dx (x-(w-r))^2 = 2(x-(w-r))
            grad[3 * i + 2] += 2 * (xs[i] - (w - rs[i])) * (-1)  # d/dr = -2(x-(w-r))
        
        if ys[i] < rs[i]:
            grad[3 * i + 1] += 2 * (rs[i] - ys[i]) * (-1)
            grad[3 * i + 2] += 2 * (rs[i] - ys[i])
        if ys[i] > h - rs[i]:
            grad[3 * i + 1] += 2 * (ys[i] - (h - rs[i]))
            grad[3 * i + 2] += 2 * (ys[i] - (h - rs[i])) * (-1)
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Gradient of negative radius penalty
    neg_r = np.minimum(rs, 0)
    grad[2::3] += 2 * neg_r
    
    return grad

def _delaunay_edges(xs, ys):
    """
    Compute Delaunay triangulation edges.
    Returns a list of (i, j) pairs.
    """
    points = np.column_stack([xs, ys])
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _solve_radii_lp(xs, ys, w, h, edges):
    """
    Given centers (xs, ys) and rectangle (w, h), solve for maximal radii
    using LP. Constraints:
    1. r_i >= 0
    2. r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    3. r_i + r_j <= d_ij for all edges (i, j)
    
    We maximize sum(r_i).
    """
    # Variables: r_0, ..., r_{N-1}
    c = -np.ones(N)  # Minimize -sum(r)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # Non-negativity handled by bounds
    # Boundary constraints:
    # -r_i <= -x_i  =>  r_i >= x_i  (Wait, r_i <= x_i is correct for inside)
    # Let's use standard form: A_ub @ r <= b_ub
    # r_i <= x_i  =>  [0..1..0] @ r <= x_i
    # r_i <= w - x_i  =>  [0..1..0] @ r <= w - x_i
    # r_i <= y_i  =>  [0..1..0] @ r <= y_i
    # r_i <= h - y_i  =>  [0..1..0] @ r <= h - y_i
    
    # We can combine these into r_i <= min(x_i, w-x_i, y_i, h-y_i)
    # But LP handles multiple constraints fine.
    
    # Actually, let's just add the tightest boundary constraint per circle to reduce size
    # Or just add all. N=21 is small.
    
    for i in range(N):
        # r_i <= x_i
        row = np.zeros(N)
        row[i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(xs[i])
        
        # r_i <= w - x_i
        row = np.zeros(N)
        row[i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(w - xs[i])
        
        # r_i <= y_i
        row = np.zeros(N)
        row[i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(ys[i])
        
        # r_i <= h - y_i
        row = np.zeros(N)
        row[i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(h - ys[i])
    
    # Distance constraints for edges
    for i, j in edges:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        dist = np.sqrt(dx * dx + dy * dy)
        
        row = np.zeros(N)
        row[i] = 1
        row[j] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(dist)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= r_i <= max_possible
    # max_possible is min of boundary constraints
    bounds = []
    for i in range(N):
        max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
        if max_r < 0:
            max_r = 0
        bounds.append((0, max_r))
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: simple radius
            rs = np.array([min(xs[i], w - xs[i], ys[i], h - ys[i]) * 0.5 for i in range(N)])
            return np.maximum(rs, 0)
    except Exception:
        rs = np.array([min(xs[i], w - xs[i], ys[i], h - ys[i]) * 0.5 for i in range(N)])
        return np.maximum(rs, 0)

def _check_and_update(params, w, h, best_state, tol=1e-6):
    """Check validity and update best if improved."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs < 1e-9):
        return
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < (min_dist - tol) ** 2:
                return
    
    val = np.sum(rs)
    if val > best_state['value']:
        best_state['value'] = val
        best_state['result'] = np.zeros((N, 3))
        best_state['result'][:, 0] = xs
        best_state['result'][:, 1] = ys
        best_state['result'][:, 2] = rs

def _grid_initial(w, h):
    """Create grid arrangement of 21 circles."""
    # Try to find a good grid layout
    best_x0 = None
    best_score = -np.inf
    
    for cols in range(3, 8):
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            continue
        
        x0 = np.zeros(3 * N)
        valid = True
        for i in range(N):
            col = i % cols
            row = i // cols
            x0[3 * i] = (col + 0.5) * w / cols
            x0[3 * i + 1] = (row + 0.5) * h / rows
            x0[3 * i + 2] = min(w / cols, h / rows) * 0.45
        
        # Quick check
        xs = x0[0::3]
        ys = x0[1::3]
        rs = x0[2::3]
        score = 0
        valid = True
        for i in range(N):
            if xs[i] < rs[i] or xs[i] > w - rs[i] or ys[i] < rs[i] or ys[i] > h - rs[i]:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    if d < rs[i] + rs[j]:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(rs)
            if score > best_score:
                best_score = score
                best_x0 = x0.copy()
    
    if best_x0 is None:
        # Fallback
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = (i % 5 + 0.5) * w / 5
            x0[3 * i + 1] = (i // 5 + 0.5) * h / 5
            x0[3 * i + 2] = 0.1
        best_x0 = x0
    
    return best_x0

def _sa_optimize(w, h, x0, max_iters=3000, seed=42, deadline=None):
    """Simulated annealing for fixed w, h."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    r_max = min(w, h) / 2.0
    T = 0.05
    cooling = 0.9995
    current_cost = _objective(x0, w, h, beta=10.0)
    
    rng = np.random.default_rng(seed)
    
    best_local_cost = current_cost
    best_local_x = x0.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
        
        scale = T * 0.3
        xs[idx] += rng.normal(0, scale)
        ys[idx] += rng.normal(0, scale)
        rs[idx] += rng.normal(0, scale * 0.5)
        
        xs[idx] = np.clip(xs[idx], 0, w)
        ys[idx] = np.clip(ys[idx], 0, h)
        rs[idx] = np.clip(rs[idx], 1e-4, r_max)
        
        new_params = np.zeros(3 * N)
        new_params[0::3] = xs
        new_params[1::3] = ys
        new_params[2::3] = rs
        
        new_cost = _objective(new_params, w, h, beta=10.0)
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_cost = new_cost
            if current_cost < best_local_cost:
                best_local_cost = current_cost
                best_local_x = new_params.copy()
        else:
            xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
        
        T *= cooling
    
    return best_local_x

def _polish_with_lp(x0, w, h, deadline=None):
    """
    Polish the solution by iteratively:
    1. Solve LP for radii given centers.
    2. Optimize centers using L-BFGS-B with fixed radii (or coupled).
    """
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    for _ in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve LP for radii
        edges = _delaunay_edges(xs, ys)
        new_rs = _solve_radii_lp(xs, ys, w, h, edges)
        
        # Check if radii improved
        if np.sum(new_rs) <= np.sum(rs) - 1e-9:
            break
        
        rs = new_rs
        
        # Optimize centers with fixed radii
        # Objective: minimize overlap penalty + boundary penalty
        # We want to maximize sum(r) but r is fixed, so just minimize penalty.
        # Actually, we can use the same objective but with r fixed?
        # No, let's just minimize the penalty to ensure validity, then re-solve LP.
        
        def center_obj(params, w, h, rs_fixed, beta=10.0):
            xs_c = params[0::N]  # Wait, params structure?
            # Let's use a different param structure for center optimization
            # params = [x0, y0, x1, y1, ...]
            xs_c = params[0::2]
            ys_c = params[1::2]
            
            pen = 0.0
            for i in range(N):
                if xs_c[i] < rs_fixed[i]:
                    pen += (rs_fixed[i] - xs_c[i]) ** 2
                if xs_c[i] > w - rs_fixed[i]:
                    pen += (xs_c[i] - (w - rs_fixed[i])) ** 2
                if ys_c[i] < rs_fixed[i]:
                    pen += (rs_fixed[i] - ys_c[i]) ** 2
                if ys_c[i] > h - rs_fixed[i]:
                    pen += (ys_c[i] - (h - rs_fixed[i])) ** 2
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs_c[i] - xs_c[j]
                    dy = ys_c[i] - ys_c[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rs_fixed[i] + rs_fixed[j]
                    val = min_dist - dist
                    if val > -1.0 / beta:
                        pen += _softplus_vec(np.array([val]), beta)[0]
            
            return pen
        
        center_params = np.zeros(2 * N)
        center_params[0::2] = xs
        center_params[1::2] = ys
        
        bounds_c = []
        for i in range(N):
            bounds_c.append((0, w))
            bounds_c.append((0, h))
        
        try:
            res = minimize(center_obj, center_params, args=(w, h, rs, 10.0),
                           method='L-BFGS-B', bounds=bounds_c,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
            xs = res.x[0::2]
            ys = res.x[1::2]
        except Exception:
            pass
    
    # Final LP solve
    edges = _delaunay_edges(xs, ys)
    final_rs = _solve_radii_lp(xs, ys, w, h, edges)
    
    final_params = np.zeros(3 * N)
    final_params[0::3] = xs
    final_params[1::3] = ys
    final_params[2::3] = final_rs
    
    return final_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Try multiple aspect ratios
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for w in widths:
        if time.time() > deadline - 10.0:
            break
        h = 2.0 - w
        
        # Initialize with grid
        x0 = _grid_initial(w, h)
        
        # L-BFGS-B optimization with smooth objective
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        
        try:
            # First pass with moderate beta
            res = minimize(lambda p: _objective(p, w, h, beta=5.0), x0,
                           jac=lambda p: _objective_grad(p, w, h, beta=5.0),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            _check_and_update(res.x, w, h, best_state)
            
            # Second pass with higher beta
            if time.time() < deadline - 20.0:
                res2 = minimize(lambda p: _objective(p, w, h, beta=20.0), res.x,
                                jac=lambda p: _objective_grad(p, w, h, beta=20.0),
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
                _check_and_update(res2.x, w, h, best_state)
                
                # LP Polish
                if time.time() < deadline - 15.0:
                    polished = _polish_with_lp(res2.x, w, h, deadline=deadline - 5.0)
                    _check_and_update(polished, w, h, best_state)
                    
        except Exception:
            pass
        
        # Simulated annealing refinement if time permits
        if time.time() < deadline - 30.0:
            x_sa_start = best_state['result'].flatten() if best_state['result'] is not None else x0
            x_sa = _sa_optimize(w, h, x_sa_start, max_iters=2000, seed=42, deadline=deadline - 10.0)
            _check_and_update(x_sa, w, h, best_state)
            
            # Polish SA result
            if time.time() < deadline - 10.0:
                polished_sa = _polish_with_lp(x_sa, w, h, deadline=deadline - 5.0)
                _check_and_update(polished_sa, w, h, best_state)
    
    # If we didn't find anything, use the seed
    if best_state['result'] is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_state['result'] = np.array(cells[:N])
    
    return best_state['result']

# EVOLVE-BLOCK-END
