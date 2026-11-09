# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_TOL = 1e-7
_DEADLINE_S = 620.0  # safety margin before 630s kill


def _feasible(centers, sides, tol=_TOL):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # inside [-tol, 1+tol]
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _grid_init(n=10):
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _nlp_objective(x, n):
    """x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]"""
    s = x[2 * n:3 * n]
    return -np.sum(s)


def _nlp_constraints(n):
    cons = []
    # boundary: cx - s/2 >= 0
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2 * n + i] / 2})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i] - x[2 * n + i] / 2})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n + i] - x[2 * n + i] / 2})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[n + i] - x[2 * n + i] / 2})
        # side > 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * n + i] - 1e-4})
    # non-overlap: for each pair, |dx| >= (si+sj)/2 OR |dy| >= (si+sj)/2
    # Model as: max(0, (si+sj)/2 - |dx|) + max(0, (si+sj)/2 - |dy|) <= 0
    # This is smooth-ish; use squared version for differentiability:
    # Actually SLSQP can handle non-smooth to some extent, but let's use a smooth surrogate:
    # g_ij = ((si+sj)/2)^2 - dx^2 - dy^2 <= 0  (this is the "both overlap" condition negated)
    # Wait: overlap iff |dx| < (si+sj)/2 AND |dy| < (si+sj)/2
    # So non-overlap: |dx| >= (si+sj)/2 OR |dy| >= (si+sj)/2
    # Equivalently: NOT(|dx| < (si+sj)/2 AND |dy| < (si+sj)/2)
    # = |dx|^2 >= ((si+sj)/2)^2 OR |dy|^2 >= ((si+sj)/2)^2
    # This is a disjunction, hard for NLP.
    #
    # Better approach: use the standard "separation" formulation with binary variables,
    # but SLSQP doesn't support binaries. Instead, we can use a penalty/soft approach
    # or reformulate.
    #
    # Alternative: for each pair, enforce that they are separated in at least one axis.
    # We can do this by adding a constraint that the "penetration depth" is zero:
    # pen_ij = max(0, (si+sj)/2 - |dx|) * max(0, (si+sj)/2 - |dy|) <= 0
    # This product is 0 iff at least one factor is 0, which is exactly non-overlap.
    # It's not smooth at 0, but SLSQP may handle it.
    #
    # Let's use the product formulation:
    for i in range(n):
        for j in range(i + 1, n):
            def sep_constraint(x, i=i, j=j):
                si = x[2 * n + i]
                sj = x[2 * n + j]
                dx = x[i] - x[j]
                dy = x[n + i] - x[n + j]
                # penetration in x: how much they overlap in x
                pen_x = (si + sj) / 2 - abs(dx)
                pen_y = (si + sj) / 2 - abs(dy)
                # We want NOT (pen_x > 0 AND pen_y > 0)
                # i.e., min(pen_x, 0) <= 0 OR min(pen_y, 0) <= 0 ... no
                # We want: pen_x <= 0 OR pen_y <= 0
                # Equivalently: min(pen_x, pen_y) <= 0  ... no that's wrong
                # pen_x <= 0 OR pen_y <= 0 is equivalent to min(pen_x, pen_y) <= 0? No.
                # If pen_x = -1, pen_y = 5: min = -1 <= 0, and pen_x <= 0 is true. OK.
                # If pen_x = 5, pen_y = -1: min = -1 <= 0, and pen_y <= 0 is true. OK.
                # If pen_x = 5, pen_y = 3: min = 3 > 0, and neither <= 0. OK (violation).
                # If pen_x = -1, pen_y = -2: min = -2 <= 0. OK.
                # So min(pen_x, pen_y) <= 0 is equivalent to (pen_x <= 0 OR pen_y <= 0).
                # But min is not smooth. Use a smooth approximation:
                # soft_min(a, b) = -log(exp(-a) + exp(-b)) / log(2) ... or use:
                # Let's just use the product: pen_x * pen_y <= 0 doesn't work either.
                # 
                # Actually, the cleanest smooth formulation:
                # g = pen_x^2 + pen_y^2 - 2*max(0,pen_x)*max(0,pen_y) ... no.
                #
                # Let me think differently. The standard approach for packing is to
                # use the constraint: (si+sj)^2/4 <= dx^2 + dy^2 is NOT correct for axis-aligned.
                #
                # For axis-aligned squares, non-overlap is:
                # |dx| >= (si+sj)/2  OR  |dy| >= (si+sj)/2
                #
                # A common NLP trick: introduce auxiliary variables and use the fact that
                # the feasible set is a union of regions. SLSQP can't handle this directly.
                #
                # Alternative approach: use a penalty method with a smooth penalty for overlap.
                # Or: use multiple local solves with different "separation patterns".
                #
                # Given complexity, let's use a different strategy:
                # 1. Start from a good grid packing.
                # 2. Use a local optimization that treats the separation pattern as fixed.
                #    For a given pattern (which axis separates each pair), the constraints
                #    are linear/quadratic and smooth.
                # 3. Try several patterns.
                #
                # For now, let's use a simpler approach: optimize with a smooth penalty
                # for overlaps, then verify feasibility.
                return 0  # placeholder
            cons.append({'type': 'ineq', 'fun': sep_constraint})
    return cons


def _optimize_from(centers, sides, n=10, max_iter=500):
    """Try to improve a feasible packing using SLSQP with penalty for overlaps."""
    x0 = np.concatenate([centers[:, 0], centers[:, 1], sides])
    
    def objective(x):
        s = x[2 * n:3 * n]
        cx = x[:n]
        cy = x[n:2 * n]
        # Penalty for boundary violations
        pen = 0.0
        for i in range(n):
            pen += max(0, -(cx[i] - s[i] / 2)) ** 2
            pen += max(0, cx[i] + s[i] / 2 - 1) ** 2
            pen += max(0, -(cy[i] - s[i] / 2)) ** 2
            pen += max(0, cy[i] + s[i] / 2 - 1) ** 2
        # Penalty for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                t = (s[i] + s[j]) / 2
                pen_x = max(0, t - abs(dx))
                pen_y = max(0, t - abs(dy))
                # Overlap if both pen_x > 0 and pen_y > 0
                # Penalty for the "overlap area" proxy
                pen += (pen_x * pen_y) ** 2
        # Penalty for side <= 0
        pen += np.sum(np.maximum(0, -s) ** 2)
        return -np.sum(s) + 1000.0 * pen
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # cx
    for i in range(n):
        bounds.append((0, 1))  # cy
    for i in range(n):
        bounds.append((1e-4, 1.0))  # s
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-12})
    
    if result.success or result.fun < objective(x0):
        cx = result.x[:n]
        cy = result.x[n:2 * n]
        s = result.x[2 * n:3 * n]
        new_centers = np.stack([cx, cy], axis=1)
        new_sides = s
        if _feasible(new_centers, new_sides):
            return new_centers, new_sides
    return centers, sides


def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    
    # Start with grid packing
    centers, sides = _grid_init(n)
    best_centers, best_sides = centers.copy(), sides.copy()
    best_score = np.sum(sides)
    
    # Try NLP optimization from the grid start
    if time.time() - start_time < _DEADLINE_S - 30:
        c_opt, s_opt = _optimize_from(centers, sides, n, max_iter=300)
        if np.sum(s_opt) > best_score:
            best_centers, best_sides = c_opt, s_opt
            best_score = np.sum(s_opt)
    
    # Try a few different initializations
    # Initialization 2: 3x3 grid with one cell split
    # Place 9 squares in a 3x3 grid, and try to fit a 10th
    # Or: 2x5 grid
    for k_cols in [2, 3, 4, 5]:
        if time.time() - start_time > _DEADLINE_S - 20:
            break
        k_rows = int(np.ceil(n / k_cols))
        side = min(1.0 / k_cols, 1.0 / k_rows)
        idx = np.arange(n)
        cols = idx % k_cols
        rows = idx // k_cols
        cx = (cols + 0.5) * (1.0 / k_cols)
        cy = (rows + 0.5) * (1.0 / k_rows)
        centers_k = np.stack([cx, cy], axis=1).astype(np.float64)
        sides_k = np.full(n, side, dtype=np.float64)
        
        if not _feasible(centers_k, sides_k):
            continue
        
        if time.time() - start_time < _DEADLINE_S - 30:
            c_opt, s_opt = _optimize_from(centers_k, sides_k, n, max_iter=200)
            if np.sum(s_opt) > best_score:
                best_centers, best_sides = c_opt, s_opt
                best_score = np.sum(s_opt)
    
    # Random restarts
    np.random.seed(42)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > _DEADLINE_S - 20:
            break
        # Random feasible-ish start: small squares scattered
        s_rand = np.random.uniform(0.05, 0.25, n)
        cx_rand = np.random.uniform(s_rand / 2, 1 - s_rand / 2, n)
        cy_rand = np.random.uniform(s_rand / 2, 1 - s_rand / 2, n)
        centers_r = np.stack([cx_rand, cy_rand], axis=1)
        
        # Check feasibility; if not, skip
        if not _feasible(centers_r, s_rand):
            continue
        
        if time.time() - start_time < _DEADLINE_S - 30:
            c_opt, s_opt = _optimize_from(centers_r, s_rand, n, max_iter=200)
            if np.sum(s_opt) > best_score:
                best_centers, best_sides = c_opt, s_opt
                best_score = np.sum(s_opt)
    
    # Final verification
    if not _feasible(best_centers, best_sides):
        # Fall back to grid
        best_centers, best_sides = _grid_init(n)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
