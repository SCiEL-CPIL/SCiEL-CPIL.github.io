"""Circle packing in unit square – 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26
SEED_TIME_LIMIT = 140.0  # wall-clock seconds before we must return


def _check_valid(arr):
    """Return True if arr is a valid packing (disjoint interiors, inside [0,1]^2)."""
    if arr is None or len(arr) != N:
        return False
    for x, y, r in arr:
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(N):
        xi, yi, ri = arr[i]
        for j in range(i + 1, N):
            xj, yj, rj = arr[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < ri + rj - 1e-9:
                return False
    return True


def _trivial_seed():
    """Fallback: simple grid packing that is always valid."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for j in range(side):
        for i in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-6
            cells.append([x, y, r])
    # We need exactly N
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.01])
    return np.array(cells[:N])


def _initial_positions(n):
    """Generate initial positions: a jittered grid."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, side), np.linspace(0.1, 0.9, side))
    xs = xs.flatten()[:n]
    ys = ys.flatten()[:n]
    # Add small random jitter
    xs = xs + np.random.uniform(-0.02, 0.02, n)
    ys = ys + np.random.uniform(-0.02, 0.02, n)
    # Clamp to [0.05, 0.95]
    xs = np.clip(xs, 0.05, 0.95)
    ys = np.clip(ys, 0.05, 0.95)
    return xs, ys


def _soft_overlap_penalty(xs, ys, rs, beta=50.0):
    """Soft penalty for overlaps using smooth approximation."""
    n = len(rs)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                gap = min_dist - dist
                penalty += np.exp(beta * gap)
    return penalty


def _build_overlap_constraints(xs, ys, rs, tol=1e-8):
    """Build linear constraints ensuring non-overlap (tangent constraints).
    
    Uses a first-order approximation: for each pair, require
    (xi - xj)*(xi - xj) + (yi - yj)*(yi - yj) >= (ri + rj)^2
    
    This is nonlinear, so we use SLSQP with nonlinear constraints.
    """
    pass


def _optimize_radii(xs, ys, n, time_left):
    """Given fixed positions, optimize radii to maximize sum while maintaining
    non-overlap and boundary constraints.
    
    Constraints:
    - rs[i] <= xs[i] (left boundary)
    - rs[i] <= 1 - xs[i] (right boundary)
    - rs[i] <= ys[i] (bottom boundary)
    - rs[i] <= 1 - ys[i] (top boundary)
    - rs[i] + rs[j] <= dist(i,j) for all i < j
    
    All linear in rs!
    """
    from scipy.optimize import linprog
    
    # Objective: minimize -sum(rs)
    c = -np.ones(n)
    
    # A_ub * rs <= b_ub
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        # rs[i] <= xs[i]
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(xs[i])
        
        # rs[i] <= 1 - xs[i]
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - xs[i])
        
        # rs[i] <= ys[i]
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(ys[i])
        
        # rs[i] <= 1 - ys[i]
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - ys[i])
    
    # Pairwise non-overlap: rs[i] + rs[j] <= dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: rs[i] >= 0
    bounds = [(0.0, None)] * n
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        rs = result.x
        # Clamp to non-negative
        rs = np.maximum(rs, 0.0)
        return rs
    else:
        # Fallback: compute max feasible radius for each circle individually
        rs = np.zeros(n)
        for i in range(n):
            min_bound = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            rs[i] = max(0.0, min_bound * 0.9)
        return rs


def _optimize_positions(rs, n, time_left, max_iters=200):
    """Optimize positions for given radii using SLSQP.
    
    Objective: minimize 0 (feasibility problem)
    Constraints: non-overlap and boundary
    """
    xs0, ys0 = _initial_positions(n)
    x0 = np.concatenate([xs0, ys0])
    
    constraints = []
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            ri = rs[i]
            rj = rs[j]
            min_dist = ri + rj
            
            def _overlap_constraint(x, i=i, j=j, min_dist=min_dist):
                xi = x[i]
                yi = x[n + i]
                xj = x[j]
                yj = x[n + j]
                dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                return dist_sq - min_dist * min_dist
            
            # Initial value
            val = _overlap_constraint(x0)
            if val < 0:
                # Only add constraint if currently violated or close
                pass
            
            constraints.append({'type': 'ineq', 'fun': _overlap_constraint})
    
    # Boundary constraints: x[i] - r[i] >= 0 => x[i] >= r[i]
    for i in range(n):
        ri = rs[i]
        
        def _left_boundary(x, i=i, ri=ri):
            return x[i] - ri
        
        def _right_boundary(x, i=i, ri=ri):
            return (1.0 - ri) - x[i]
        
        def _bottom_boundary(x, i=i, ri=ri):
            return x[n + i] - ri
        
        def _top_boundary(x, i=i, ri=ri):
            return (1.0 - ri) - x[n + i]
        
        constraints.append({'type': 'ineq', 'fun': _left_boundary})
        constraints.append({'type': 'ineq', 'fun': _right_boundary})
        constraints.append({'type': 'ineq', 'fun': _bottom_boundary})
        constraints.append({'type': 'ineq', 'fun': _top_boundary})
    
    bounds = [(0.0, 1.0)] * (2 * n)
    
    objective = lambda x: 0.0
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iters,
                                                           'ftol': 1e-10})
        if result.success:
            xs_opt = result.x[:n]
            ys_opt = result.x[n:]
            return xs_opt, ys_opt
        else:
            return xs0, ys0
    except Exception:
        return xs0, ys0


def _iterative_packing(n, time_budget):
    """Iterative approach:
    1. Start with equal radii
    2. Optimize positions for fixed radii
    3. Optimize radii for fixed positions (LP)
    4. Repeat until convergence or time runs out
    """
    start_time = time.time()
    
    # Initial radii
    side = int(np.ceil(np.sqrt(n)))
    r_init = 0.5 / side * 0.8
    rs = np.full(n, r_init)
    
    best_rs = rs.copy()
    best_sum = rs.sum()
    best_xs, best_ys = _initial_positions(n)
    
    for iteration in range(50):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Step 1: Optimize positions for fixed radii
        xs, ys = _optimize_positions(rs, n, time_budget - elapsed, max_iters=100)
        
        # Step 2: Optimize radii for fixed positions
        rs_new = _optimize_radii(xs, ys, n, time_budget - elapsed)
        
        new_sum = rs_new.sum()
        if new_sum > best_sum - 1e-12:
            best_sum = new_sum
            best_rs = rs_new.copy()
            best_xs = xs.copy()
            best_ys = ys.copy()
        
        # Check convergence
        if abs(new_sum - rs.sum()) < 1e-10:
            break
        
        rs = rs_new
    
    return best_xs, best_ys, best_rs, best_sum


def construct():
    """Pack 26 circles with disjoint interiors inside [0,1]^2 maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + SEED_TIME_LIMIT
    
    np.random.seed(42)
    
    # Try multiple random restarts
    best_result = None
    best_sum = -np.inf
    
    # Get initial positions and try different seeds
    n_restarts = 10
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:  # Need at least 20s for finishing
            break
        
        np.random.seed(42 + restart * 137)
        
        # Run iterative packing
        try:
            xs, ys, rs, total = _iterative_packing(N, deadline - time.time() - 5)
            
            # Build result array
            result = np.column_stack([xs, ys, rs])
            
            # Validate
            if _check_valid(result):
                if total > best_sum:
                    best_sum = total
                    best_result = result
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to trivial seed
        return _trivial_seed()


# Keep the required function name
construct.__doc__ = "Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii."
