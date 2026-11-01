"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay


def _objective(params, n):
    """
    Vectorized objective: minimize -sum(r) + penalty for violations.
    Uses smooth quadratic penalties for boundary and overlap constraints.
    """
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.maximum(pts[:, 2], 1e-8)

    sum_r = np.sum(r)

    # Boundary penalties (vectorized)
    # For each circle: violations of x >= r, x <= 1-r, y >= r, y <= 1-r
    viol = np.zeros((n, 4))
    viol[:, 0] = np.maximum(0, r - x)           # x < r
    viol[:, 1] = np.maximum(0, r - (1.0 - x))   # x > 1-r
    viol[:, 2] = np.maximum(0, r - y)           # y < r
    viol[:, 3] = np.maximum(0, r - (1.0 - y))   # y > 1-y
    bound_pen = np.sum(viol * viol)

    # Overlap penalties (vectorized via broadcasting)
    # dist[i,j] = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-16)  # avoid div by zero

    min_dist = r[:, None] + r[None, :]
    viol_overlap = np.maximum(0, min_dist - dist)
    # Only count upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    overlap_pen = np.sum(viol_overlap[iu] * viol_overlap[iu])

    total_pen = bound_pen + overlap_pen
    return -sum_r + 10.0 * total_pen


def _objective_grad(params, n):
    """
    Analytic gradient of the objective.
    """
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.maximum(pts[:, 2], 1e-8)

    grad = np.zeros_like(params)

    # d(-sum_r)/dr_i = -1
    grad[2::3] += -1.0

    # Boundary penalty gradients
    # viol[:, 0] = max(0, r - x)  -> d/dx = -1 if r-x>0, d/dr = 1 if r-x>0
    # viol[:, 1] = max(0, r - (1-x)) = max(0, r + x - 1) -> d/dx = 1 if r+x-1>0, d/dr = 1
    # viol[:, 2] = max(0, r - y)  -> d/dy = -1 if r-y>0, d/dr = 1
    # viol[:, 3] = max(0, r - (1-y)) = max(0, r + y - 1) -> d/dy = 1 if r+y-1>0, d/dr = 1

    m0 = (r - x) > 0
    m1 = (r + x - 1.0) > 0
    m2 = (r - y) > 0
    m3 = (r + y - 1.0) > 0

    # Penalty = sum of (viol)^2, so d(pen)/d(var) = 2*viol * d(viol)/d(var)
    # grad_x contribution from bound_pen:
    # d/dx [ (r-x)^2 ] = 2(r-x)(-1) = -2(r-x) when m0
    # d/dx [ (r+x-1)^2 ] = 2(r+x-1)(1) = 2(r+x-1) when m1
    grad[0::3] += 10.0 * (-2.0 * (r - x) * m0 + 2.0 * (r + x - 1.0) * m1)

    # grad_y contribution:
    grad[1::3] += 10.0 * (-2.0 * (r - y) * m2 + 2.0 * (r + y - 1.0) * m3)

    # grad_r contribution from boundary:
    # d/dr [(r-x)^2] = 2(r-x) when m0
    # d/dr [(r+x-1)^2] = 2(r+x-1) when m1
    # d/dr [(r-y)^2] = 2(r-y) when m2
    # d/dr [(r+y-1)^2] = 2(r+y-1) when m3
    grad[2::3] += 10.0 * (2.0 * (r - x) * m0 + 2.0 * (r + x - 1.0) * m1 +
                           2.0 * (r - y) * m2 + 2.0 * (r + y - 1.0) * m3)

    # Overlap penalty gradients
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-16)

    min_dist = r[:, None] + r[None, :]
    viol = np.maximum(0, min_dist - dist)

    # For each pair (i,j) with i<j, penalty contribution = (min_dist_ij - dist_ij)^2 if positive
    # d/dx_i: 2*viol * (-(x_i - x_j)/dist) = -2*viol*dx_ij/dist_ij
    # d/dx_j: 2*viol * (-(x_j - x_i)/dist) = -2*viol*dx_ji/dist_ij = 2*viol*dx_ij/dist_ij
    # d/dr_i: 2*viol * 1 = 2*viol
    # d/dr_j: 2*viol * 1 = 2*viol

    # Use only upper triangle
    iu = np.triu_indices(n, k=1)
    v = viol[iu]  # violations for upper triangle pairs
    d = dist[iu]  # distances for upper triangle pairs
    dx_u = dx[iu]  # x_i - x_j for i<j
    dy_u = dy[iu]  # y_i - y_j for i<j

    # Gradient contributions for each pair
    # For pair (i,j):
    #   d/dx_i = -2*v*dx_u/d,  d/dx_j = +2*v*dx_u/d
    #   d/dy_i = -2*v*dy_u/d,  d/dy_j = +2*v*dy_u/d
    #   d/dr_i = 2*v,           d/dr_j = 2*v

    idx_i = iu[0]
    idx_j = iu[1]

    # Accumulate gradients using np.add.at for repeated indices
    gx_i = -2.0 * v * dx_u / (d + 1e-16)
    gx_j = 2.0 * v * dx_u / (d + 1e-16)
    gy_i = -2.0 * v * dy_u / (d + 1e-16)
    gy_j = 2.0 * v * dy_u / (d + 1e-16)
    gr = 2.0 * v


    # Actually, let's do it properly:
    # Reset and recompute overlap grad contributions
    # We need to accumulate into grad[0::3], grad[1::3], grad[2::3]

    # Create temporary arrays for accumulation
    grad_x_ov = np.zeros(n)
    grad_y_ov = np.zeros(n)
    grad_r_ov = np.zeros(n)

    np.add.at(grad_x_ov, idx_i, gx_i)
    np.add.at(grad_x_ov, idx_j, gx_j)
    np.add.at(grad_y_ov, idx_i, gy_i)
    np.add.at(grad_y_ov, idx_j, gy_j)
    np.add.at(grad_r_ov, idx_i, gr)
    np.add.at(grad_r_ov, idx_j, gr)

    grad[0::3] += 10.0 * grad_x_ov
    grad[1::3] += 10.0 * grad_y_ov
    grad[2::3] += 10.0 * grad_r_ov

    return grad


def _is_valid(pts, n, tol=1e-9):
    """Check if all circles are inside unit square and non-overlapping."""
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    # Boundary check
    if np.any(r < -tol) or np.any(x < r - tol) or np.any(x > 1 - r - tol) \
       or np.any(y < r - tol) or np.any(y > 1 - r - tol):
        return False
    # Overlap check (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-16)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True


def _sum_radii(pts):
    return np.sum(pts[:, 2])


def _grid_init(n, jitter=0.0):
    """Create a grid of n circles with optional jitter."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            if jitter > 0:
                cx += np.random.uniform(-jitter, jitter)
                cy += np.random.uniform(-jitter, jitter)
                cx = np.clip(cx, r, 1 - r)
                cy = np.clip(cy, r, 1 - r)
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _voronoi_init(n, n_points=200):
    """
    Generate initial configuration using Voronoi tessellation.
    Place points randomly, compute Voronoi cells, inscribe circles.
    """
    # Generate random points
    np.random.seed(42)
    pts = np.random.uniform(0.05, 0.95, (n_points, 2))

    try:
        vor = Voronoi(pts)
        # For each of the n target circles, pick a random Voronoi vertex region
        # Simpler: use the n_points as centers, compute inradii
        # For each point, the inradius is the min distance to its Voronoi edges
        # But that's complex. Instead, use a simpler approach:
        # Use Delaunay to find neighbors, then estimate max circle radius
    except Exception:
        pass

    # Fallback: use grid with more jitter
    return _grid_init(n, jitter=0.08)


def _optimize_single(pts, n, deadline, method='L-BFGS-B'):
    """
    Optimize a single starting configuration using analytic gradient.
    Returns (best_pts, best_value, success).
    """
    params = pts.flatten().copy()

    try:
        res = minimize(
            _objective,
            params,
            jac=_objective_grad,
            args=(n,),
            method=method,
            options={'maxiter': 300, 'gtol': 1e-8, 'disp': False, 'maxfun': 2000}
        )
        if time.time() > deadline - 2:
            return None, None, False

        res_pts = res.x.reshape(n, 3)
        # Clip to valid range
        res_pts[:, 0] = np.clip(res_pts[:, 0], 0, 1)
        res_pts[:, 1] = np.clip(res_pts[:, 1], 0, 1)
        res_pts[:, 2] = np.maximum(res_pts[:, 2], 0)

        # Enforce boundary constraints
        for i in range(n):
            r = res_pts[i, 2]
            res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1 - r)
            res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1 - r)
            max_r = min(res_pts[i, 0], 1 - res_pts[i, 0], res_pts[i, 1], 1 - res_pts[i, 1])
            if res_pts[i, 2] > max_r:
                res_pts[i, 2] = max_r

        if _is_valid(res_pts, n):
            return res_pts, _sum_radii(res_pts), True
        else:
            # Try to repair by shrinking radii
            for i in range(n):
                r = res_pts[i, 2]
                res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1 - r)
                res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1 - r)
                max_r = min(res_pts[i, 0], 1 - res_pts[i, 0], res_pts[i, 1], 1 - res_pts[i, 1])
                if res_pts[i, 2] > max_r:
                    res_pts[i, 2] = max_r
            if _is_valid(res_pts, n):
                return res_pts, _sum_radii(res_pts), True
    except Exception:
        pass

    return None, None, False


def _polish(pts, n, deadline):
    """
    High-precision polish pass on an already-good configuration.
    Uses tighter tolerances and more iterations.
    """
    params = pts.flatten().copy()
    try:
        res = minimize(
            _objective,
            params,
            jac=_objective_grad,
            args=(n,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-10, 'disp': False, 'maxfun': 5000}
        )
        if time.time() > deadline - 1:
            return None, None

        res_pts = res.x.reshape(n, 3)
        res_pts[:, 0] = np.clip(res_pts[:, 0], 0, 1)
        res_pts[:, 1] = np.clip(res_pts[:, 1], 0, 1)
        res_pts[:, 2] = np.maximum(res_pts[:, 2], 0)

        for i in range(n):
            r = res_pts[i, 2]
            res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1 - r)
            res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1 - r)
            max_r = min(res_pts[i, 0], 1 - res_pts[i, 0], res_pts[i, 1], 1 - res_pts[i, 1])
            if res_pts[i, 2] > max_r:
                res_pts[i, 2] = max_r

        if _is_valid(res_pts, n):
            return res_pts, _sum_radii(res_pts)
    except Exception:
        pass

    return None, None


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin

    best_pts = None
    best_value = -np.inf

    # Strategy: multiple restarts with varied initializations
    n_restarts = 25

    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break

        # Vary initialization strategy
        if restart == 0:
            # Clean grid
            pts = _grid_init(N, jitter=0.0)
        elif restart == 1:
            # Small jitter
            pts = _grid_init(N, jitter=0.02)
        elif restart == 2:
            # Medium jitter
            pts = _grid_init(N, jitter=0.05)
        else:
            # Random perturbation of grid
            jitter = 0.03 + 0.07 * np.random.random()
            pts = _grid_init(N, jitter=jitter)
            # Additional random perturbation
            pts[:, 0] += np.random.uniform(-0.08, 0.08, N)
            pts[:, 1] += np.random.uniform(-0.08, 0.08, N)
            pts[:, 2] *= np.random.uniform(0.85, 1.0, N)
            # Clip to valid range
            for i in range(N):
                r = pts[i, 2]
                pts[i, 0] = np.clip(pts[i, 0], r, 1 - r)
                pts[i, 1] = np.clip(pts[i, 1], r, 1 - r)

        # Optimize
        res_pts, res_val, success = _optimize_single(pts, N, deadline)

        if success and res_val > best_value:
            best_value = res_val
            best_pts = res_pts.copy()

    # Final polish: high-precision pass on the best solution
    if best_pts is not None and time.time() < deadline - 2:
        polished_pts, polished_val = _polish(best_pts, N, deadline)
        if polished_pts is not None and polished_val > best_value:
            best_value = polished_val
            best_pts = polished_pts

    if best_pts is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_pts = np.array(cells[:N])

    return best_pts
# EVOLVE-BLOCK-END
