"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
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


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    # bary has shape (n, 3) or we work with (n, 2) where w = 1-u-v
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

    # Better initial guess: symmetric distribution
    # 3 corners, 3 edge midpoints, 1 center, 3 more spread
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [0.5, S3H/3],        # center-ish
        [0.333, S3H/6],      # lower left
        [0.667, S3H/6],      # lower right
        [0.5, 2*S3H/3],      # upper
        [0.2, 0.2],
    ], dtype=float)
    _update_best(init)

    def _de_objective(x_flat):
        """Objective for Differential Evolution in barycentric coords."""
        u = x_flat[0:N]
        v = x_flat[N:2*N]
        # Enforce barycentric constraints: u>=0, v>=0, u+v<=1
        # DE works in a box. We map the box [0,1]x[0,1] to the triangle.
        # Actually, simpler: DE in Cartesian with projection is easier to set up bounds for.
        # Let's stick to Cartesian with projection for DE, as bounds are [0,1]x[0,S3H] roughly.
        # But projection is non-differentiable. DE doesn't care.
        pts = x_flat.reshape(-1, 2)
        pts = _project_to_triangle(pts)
        return -_min_triangle_area(pts)

    # Phase A: Differential Evolution
    # Bounds for Cartesian coordinates:
    # x: [0, 1], y: [0, S3H]
    # We will use DE in Cartesian space.
    bounds = [(0.0, 1.0), (0.0, S3H)] * N
    pop_size = 30
    max_iter = 200
    
    # Use a subset of time for DE
    de_deadline = start_time + 100
    
    try:
        # scipy.optimize.differential_evolution
        res_de = minimize(
            _de_objective,
            init.flatten(),
            method='differential_evolution',
            bounds=bounds,
            maxiter=max_iter,
            popsize=pop_size,
            tol=1e-8,
            polish=False, # We will do polish ourselves
            init='latinhypercube',
            seed=12345,
            update_interval=10,
            callback=None
        )
        # Note: minimize with DE doesn't have a time limit callback in older SciPy versions easily.
        # We rely on maxiter and popsize to be reasonable.
        # 11 points * 2 coords = 22 dimensions.
        # popsize 30 * maxiter 200 = 6000 evaluations.
        # Each eval is vectorized min-area. Should be fast enough.
        
        best_de_pts = _project_to_triangle(res_de.x.reshape(-1, 2))
        _update_best(best_de_pts)
        
    except Exception as e:
        pass

    # Phase B: Local Polish using L-BFGS-B on the best found so far
    if best_pts is not None and _elapsed_ok(20):
        # Try a few random perturbations around the best point and polish
        for i in range(5):
            if not _elapsed_ok(10):
                break
            # Perturb best point
            cand = best_pts + rng.normal(0, 0.05, size=(N, 2))
            cand = _project_to_triangle(cand)
            
            # Optimize in barycentric coords for better constraint handling
            bary0 = _cartesian_to_barycentric(cand)
            x0 = np.concatenate([bary0[:, 0], bary0[:, 1]])
            
            def obj_l_bfgs(x):
                u = x[0:N]
                v = x[N:2*N]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1 - u)
                pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                return -_soft_min_area(pts, 500.0)
            
            try:
                res = minimize(obj_l_bfgs, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                u = np.clip(res.x[0:N], 0, 1)
                v = np.clip(res.x[N:2*N], 0, 1 - u)
                pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                _update_best(pts)
            except Exception:
                pass

    # Phase C: Final Nelder-Mead polish on the absolute best
    if best_pts is not None and _elapsed_ok(10):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_soft_min_area(p, 1000.0)
        
        try:
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        return pts

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
