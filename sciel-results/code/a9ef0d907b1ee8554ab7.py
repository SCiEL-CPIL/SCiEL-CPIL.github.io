"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIP = len(_TRIPLETS)  # 165


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v >= 0."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.array([x, y])


def _solve_socp_linear(pts_init, t_init):
    """
    Solve the linear program: maximize t subject to all triangle areas >= t,
    points inside the triangle.
    
    Variables: x_i, y_i for i in 0..10, and t. Total 23 variables.
    Objective: maximize t -> minimize -t.
    
    Constraints:
    - Point i inside triangle: y_i >= 0, x_i >= y_i/sqrt(3), x_i <= 1 - y_i/sqrt(3)
    - Triangle area constraint: |cross| >= 2t for each triplet.
      Since area = |cross|/2 >= t means |cross| >= 2t.
      The cross product can be positive or negative, so we need:
        cross >= 2t OR cross <= -2t
      This is non-linear. Instead, we use the fact that for a fixed ordering
      of points, the sign is fixed. We'll use the absolute value trick:
      Introduce auxiliary variables for each triangle.
      
      Actually, let's use a different approach. The area of triangle (i,j,k) is:
        0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
      
      This is the absolute value of a linear function. To handle |f| >= 2t,
      we can split into f >= 2t or f <= -2t. Since we don't know the sign,
      we use the standard LP trick: introduce variable z_m for each constraint m,
      with f_m >= z_m, f_m <= -z_m, z_m >= 2t. But that's not right either.
      
      Better: for each triangle m, let f_m(x,y) be the linear expression.
      We need |f_m| >= 2t. This is equivalent to f_m >= 2t or f_m <= -2t.
      In LP, we can't do OR directly. 
      
      Alternative: since we have a good initial point, we can determine the sign
      of each cross product at the initial point and fix it. This works locally.
    """
    # Determine sign of each cross product at initial point
    areas = _all_triangle_areas(pts_init)
    pi = pts_init[_T_I]
    pj = pts_init[_T_J]
    pk = pts_init[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    signs = np.sign(cross)
    signs[signs == 0] = 1.0  # default to positive if zero

    # Variables: [x_0, y_0, x_1, y_1, ..., x_10, y_10, t] -> 23 vars
    # Objective: minimize -t -> c = [0, 0, 0, 0, ..., 0, -1]
    c = np.zeros(2 * N + 1)
    c[-1] = -1.0  # maximize t

    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Point constraints: y_i >= 0 -> -y_i <= 0
    #    x_i - y_i/sqrt(3) >= 0 -> -x_i + y_i/sqrt(3) <= 0
    #    x_i + y_i/sqrt(3) <= 1 -> x_i + y_i/sqrt(3) <= 1
    A_ub = []
    b_ub = []

    for i in range(N):
        # -y_i <= 0
        row = np.zeros(2 * N + 1)
        row[2 * i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        # -x_i + y_i/sqrt(3) <= 0
        row = np.zeros(2 * N + 1)
        row[2 * i] = -1.0
        row[2 * i + 1] = 1.0 / S3
        A_ub.append(row)
        b_ub.append(0.0)

        # x_i + y_i/sqrt(3) <= 1
        row = np.zeros(2 * N + 1)
        row[2 * i] = 1.0
        row[2 * i + 1] = 1.0 / S3
        A_ub.append(row)
        b_ub.append(1.0)

    # 2. Triangle area constraints: sign_m * cross_m >= 2t
    #    cross_m = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
    #    We need: sign_m * (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)) >= 2t
    #    Rearranged: -sign_m*(...) + 2t <= 0
    for m in range(N_TRIP):
        i, j, k = _TRIPLETS[m]
        s = signs[m]
        row = np.zeros(2 * N + 1)
        # coefficient of x_i: -s*(y_j - y_k) but y_j,y_k are variables, so this is bilinear!
        # Wait, the cross product involves products of coordinates, so it's NOT linear in x,y.
        # This approach fails because the area is not linear in the point coordinates.
        # 
        # Let me reconsider. The area formula is:
        # A = 0.5 * |det([x_j-x_i, y_j-y_i; x_k-x_i, y_k-y_i])|
        #   = 0.5 * |(x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)|
        # This is quadratic in the coordinates, not linear.
        #
        # So SOCP/LP won't work directly. Let me use a different strategy.
        pass

    # Since the area is not linear, let's abandon the LP approach and use a better
    # gradient-based approach with the soft-min objective.
    return None


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Model: initialize a plausible parameterization (corners, mid-edges, centroid) ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        return _project_to_triangle(pts)

    def _optimize_from(pts0, beta, maxiter=500):
        """Optimize in barycentric coordinates with L-BFGS-B (naturally inside triangle)."""
        # Convert to barycentric (u, v) where w = 1-u-v
        bary0 = _cartesian_to_barycentric(pts0)

        def objective(x):
            u = x[0:N]
            v = x[N:2*N]
            # Clip to valid barycentric range: u>=0, v>=0, u+v<=1
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1 - u)
            pts = _barycentric_to_cartesian(np.column_stack([u, v]))
            return -_soft_min_area(pts, beta)

        try:
            x0 = np.concatenate([bary0[:, 0], bary0[:, 1]])
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12})
            u = np.clip(res.x[0:N], 0, 1)
            v = np.clip(res.x[N:2*N], 0, 1 - u)
            pts = _barycentric_to_cartesian(np.column_stack([u, v]))
            return pts
        except Exception:
            return pts0

    def _nudge_search(pts, n_iters=500):
        """
        Local nudge search: find the smallest triangle, then nudge one of its
        points in a direction that increases its area.
        """
        pts = pts.copy()
        current_area = _min_triangle_area(pts)
        
        for it in range(n_iters):
            if not _elapsed_ok(10):
                break
            
            areas = _all_triangle_areas(pts)
            min_idx = np.argmin(areas)
            i, j, k = _TRIPLETS[min_idx]
            
            # Try nudging each of the three points in multiple directions
            best_nudge = None
            best_new_area = current_area
            
            for idx in [i, j, k]:
                # Try small perturbations in random directions
                for _ in range(20):
                    direction = rng.normal(0, 1, size=2)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    for step in [0.001, 0.005, 0.01, 0.02]:
                        new_pts = pts.copy()
                        new_pts[idx] += step * direction
                        new_pts = _project_to_triangle(new_pts)
                        new_area = _min_triangle_area(new_pts)
                        if new_area > best_new_area:
                            best_new_area = new_area
                            best_nudge = new_pts.copy()
            
            if best_nudge is not None:
                pts = best_nudge
                current_area = best_new_area
            else:
                # Try larger random perturbations
                for _ in range(5):
                    new_pts = pts + rng.normal(0, 0.01, size=(N, 2))
                    new_pts = _project_to_triangle(new_pts)
                    new_area = _min_triangle_area(new_pts)
                    if new_area > current_area:
                        pts = new_pts
                        current_area = new_area
                        break
                else:
                    # Stuck, try a bigger jump
                    new_pts = pts + rng.normal(0, 0.05, size=(N, 2))
                    new_pts = _project_to_triangle(new_pts)
                    new_area = _min_triangle_area(new_pts)
                    if new_area > current_area * 0.9:  # accept slightly worse to escape
                        pts = new_pts
                        current_area = new_area
        
        return pts

    # ---- Phase A: multi-start soft-min gradient optimization (annealed beta) ----
    n_starts = 30
    bases = [init]
    for _ in range(15):
        bases.append(_start_from_base(init, 0.25))
    for _ in range(10):
        rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        bases.append(rand_pts)

    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.05)
        _update_best(pts)
        for beta in [10.0, 30.0, 80.0, 200.0, 500.0, 1200.0, 3000.0]:
            if not _elapsed_ok(40):
                break
            pts = _optimize_from(pts, beta, maxiter=300)
            _update_best(pts)

    # ---- Phase B: Nudge search on the best configuration ----
    if best_pts is not None and _elapsed_ok(30):
        pts = _nudge_search(best_pts.copy(), n_iters=300)
        _update_best(pts)
        
        # Do nudge search from a few other good starting points
        for s in range(3):
            if not _elapsed_ok(20):
                break
            base = bases[s % len(bases)]
            pts = _start_from_base(base, 0.02)
            pts = _optimize_from(pts, 500.0, maxiter=200)
            pts = _nudge_search(pts, n_iters=200)
            _update_best(pts)

    # ---- Phase C: Nelder-Mead local polish on the best configuration ----
    if best_pts is not None and _elapsed_ok(15):
        for scale in [0.03, 0.01, 0.005]:
            if not _elapsed_ok(10):
                break
            for _ in range(5):
                if not _elapsed_ok(10):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)

                def nm_obj(x):
                    p = x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    return -_soft_min_area(p, 2000.0)
                try:
                    res = minimize(nm_obj, cand.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13})
                    p = _project_to_triangle(res.x.reshape(-1, 2))
                    _update_best(p)
                except Exception:
                    pass

    # ---- Final fine polish on the absolute best ----
    if best_pts is not None and _elapsed_ok(8):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_soft_min_area(p, 5000.0)
        try:
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15})
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

    # ---- Final nudge polish ----
    if best_pts is not None and _elapsed_ok(5):
        pts = _nudge_search(best_pts.copy(), n_iters=100)
        _update_best(pts)

    if best_pts is None:
        # fallback random feasible configuration
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        return pts

    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
