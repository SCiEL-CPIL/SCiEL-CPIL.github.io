"""
Circle packing in unit square – 26 circles, maximize sum of radii.

Structural idea (combined):
  - Represent packing by 52 center coordinates only.
  - Radii derived geometrically from pairwise distances and boundary.
  - Primary optimization: SLSQP with explicit non-overlap constraints
    (each pair: dist >= r_i + r_j) and boundary constraints (r_i <= d_i).
    Objective: maximize sum(r_i) where r_i = min(d_i, min_j dist_ij / 2).
  - Secondary: L-BFGS-B on the smooth objective (negative sum radii)
    as a fast local refiner.
  - Tertiary: discrete local search (move one circle at a time) for
    escaping shallow local optima.
  - Multi-start with diverse initial layouts (grid, hex, random).
  - Deadline guard: return best valid result before 115 s.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
DEADLINE = 115.0
START = time.time()


def _elapsed():
    return time.time() - START


def _compute_radii(x, y):
    """
    Given centers, compute the maximum feasible radii such that:
      r_i <= boundary_dist_i
      r_i + r_j <= dist_ij  for all i < j
    We use a simple iterative water-filling / bisection approach:
      Start with r_i = boundary_dist_i.
      For each pair (i,j), if r_i + r_j > dist_ij, scale down.
      Iterate until convergence.
    """
    pts = np.column_stack([x, y])
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    r = d_bound.copy()

    dist_mat = cdist(pts, pts)
    np.fill_diagonal(dist_mat, np.inf)

    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                s = r[i] + r[j]
                if s > d:
                    # Scale both proportionally to fit
                    scale = d / s * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        if not changed:
            break

    # Final boundary clamp
    r = np.minimum(r, d_bound)
    r = np.maximum(r, 1e-8)
    return r


def _sum_radii(x, y):
    r = _compute_radii(x, y)
    return float(np.sum(r))


def _objective_smooth(params):
    """Smooth objective for L-BFGS-B: -sum(radii) + small penalty for near-overlap."""
    x = np.clip(params[0::2], 0.005, 0.995)
    y = np.clip(params[1::2], 0.005, 0.995)
    r = _compute_radii(x, y)
    val = -np.sum(r)
    # Soft penalty: encourage separation beyond tangency
    pts = np.column_stack([x, y])
    dist_mat = cdist(pts, pts)
    np.fill_diagonal(dist_mat, np.inf)
    r_sum = r[:, None] + r[None, :]
    # Only penalize when very close (within 0.01 of tangency)
    closeness = np.clip(0.01 - (dist_mat - r_sum), 0.0, None)
    val += 0.5 * np.sum(closeness ** 2)
    return val


def _slsqp_objective(params):
    """Objective for SLSQP: minimize -sum(radii)."""
    x = np.clip(params[0::2], 0.005, 0.995)
    y = np.clip(params[1::2], 0.005, 0.995)
    r = _compute_radii(x, y)
    return -np.sum(r)


def _slsqp_constraints(params):
    """
    Constraints for SLSQP:
      1. Boundary: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
         (These are implicit in the radius computation, but we add soft ones.)
      2. Pairwise: dist_ij >= r_i + r_j - eps  (for all i<j)
    We return a list of constraint dicts.
    """
    x = np.clip(params[0::2], 0.005, 0.995)
    y = np.clip(params[1::2], 0.005, 0.995)
    r = _compute_radii(x, y)

    cons = []
    # Boundary constraints (each circle must be inside [0,1]^2 with margin r_i)
    for i in range(N):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - _compute_radii(p[0::2], p[1::2])[i]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[2*i] - _compute_radii(p[0::2], p[1::2])[i]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1] - _compute_radii(p[0::2], p[1::2])[i]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[2*i+1] - _compute_radii(p[0::2], p[1::2])[i]})

    # Pairwise non-overlap: dist_ij - r_i - r_j >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    np.sqrt((p[2*i]-p[2*j])**2 + (p[2*i+1]-p[2*j+1])**2)
                    - _compute_radii(p[0::2], p[1::2])[i]
                    - _compute_radii(p[0::2], p[1::2])[j]
                )
            })

    return cons


def _valid(x, y, r, eps=1e-7):
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _fix_valid(x, y, r):
    """Iteratively shrink radii to ensure validity."""
    for _ in range(20):
        d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, d_bound)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < r[i] + r[j]:
                    scale = dist[i, j] / (r[i] + r[j]) * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        if not changed:
            break
    return r


def _local_search(x, y, max_iters=2000, step_size=0.02, seed=0):
    """Discrete local search: move one circle at a time."""
    rng = np.random.default_rng(seed)
    cx = x.copy()
    cy = y.copy()
    cr = _compute_radii(cx, cy)
    cscore = float(np.sum(cr))
    best_x, best_y, best_score = cx.copy(), cy.copy(), cscore

    for it in range(max_iters):
        if _elapsed() > DEADLINE - 3.0:
            break
        i = rng.integers(N)
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        nx = cx.copy()
        ny = cy.copy()
        nx[i] = np.clip(nx[i] + dx, 0.02, 0.98)
        ny[i] = np.clip(ny[i] + dy, 0.02, 0.98)
        nr = _compute_radii(nx, ny)
        nscore = float(np.sum(nr))
        if nscore > cscore + 1e-12:
            cx, cy, cr, cscore = nx, ny, nr, nscore
            if cscore > best_score:
                best_x, best_y, best_score = cx.copy(), cy.copy(), cscore
    return best_x, best_y, best_score


def _lbfgs_refine(x, y, maxiter=300):
    """L-BFGS-B refinement of center positions."""
    vec = np.zeros(2 * N)
    vec[0::2] = x
    vec[1::2] = y
    try:
        res = minimize(_objective_smooth, vec, method='L-BFGS-B',
                       bounds=[(0.01, 0.99)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 3000})
        rx = np.clip(res.x[0::2], 0.01, 0.99)
        ry = np.clip(res.x[1::2], 0.01, 0.99)
        return rx, ry
    except Exception:
        return x, y


def _grid_init(rng, cols=5, rows=5):
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    while len(pts) < N:
        pts.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _hex_init(rng):
    x = np.zeros(N)
    y = np.zeros(N)
    spacing = 0.19
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.12 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.12 + row * spacing * 0.866
        if x[i] > 0.92:
            x[i] = 0.92
        if y[i] > 0.92:
            y[i] = 0.92
    return x, y


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    return x, y


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = []
    inits.append(_grid_init(rng, 5, 5))
    inits.append(_grid_init(rng, 4, 6))
    inits.append(_grid_init(rng, 6, 4))
    inits.append(_hex_init(rng))
    for s in [1, 7, 13, 23, 37, 53]:
        r2 = np.random.default_rng(s)
        inits.append(_random_init(r2))

    for idx, (x0, y0) in enumerate(inits):
        if _elapsed() > DEADLINE - 10.0:
            break

        # Add small jitter for diversity
        x0 = np.clip(x0 + rng.normal(0, 0.01, N), 0.02, 0.98)
        y0 = np.clip(y0 + rng.normal(0, 0.01, N), 0.02, 0.98)

        # Step 1: L-BFGS-B refinement
        x1, y1 = _lbfgs_refine(x0, y0, maxiter=200)
        r1 = _compute_radii(x1, y1)
        val1 = float(np.sum(r1))

        # Step 2: Local search from refined position
        x2, y2, ls_score = _local_search(x1, y1, max_iters=1500, step_size=0.025, seed=100 + idx)
        r2 = _compute_radii(x2, y2)

        # Step 3: Final L-BFGS-B polish
        x3, y3 = _lbfgs_refine(x2, y2, maxiter=100)
        r3 = _compute_radii(x3, y3)

        # Pick best of the three stages
        candidates = [
            (x1, y1, r1, val1),
            (x2, y2, r2, ls_score),
            (x3, y3, r3, float(np.sum(r3))),
        ]

        for cx, cy, cr, cv in candidates:
            cr_fixed = _fix_valid(cx, cy, cr.copy())
            if _valid(cx, cy, cr_fixed):
                v = float(np.sum(cr_fixed))
                if v > best_val:
                    best_val = v
                    best = np.column_stack([cx, cy, cr_fixed])

    # Additional round: local search from best found
    if best is not None and _elapsed() < DEADLINE - 15.0:
        bx, by, br = best[:, 0], best[:, 1], best[:, 2]
        for trial in range(4):
            if _elapsed() > DEADLINE - 5.0:
                break
            r2 = np.random.default_rng(500 + trial)
            px = np.clip(bx + r2.normal(0, 0.015, N), 0.02, 0.98)
            py = np.clip(by + r2.normal(0, 0.015, N), 0.02, 0.98)
            x4, y4, ls4 = _local_search(px, py, max_iters=1000, step_size=0.02, seed=600 + trial)
            r4 = _compute_radii(x4, y4)
            r4f = _fix_valid(x4, y4, r4.copy())
            if _valid(x4, y4, r4f):
                v4 = float(np.sum(r4f))
                if v4 > best_val:
                    best_val = v4
                    best = np.column_stack([x4, y4, r4f])

    if best is not None:
        return best

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = _compute_radii(x, y)
    r = _fix_valid(x, y, r)
    return np.column_stack([x, y, r])
