"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, LinearConstraint, differential_evolution

N = 11
H = np.sqrt(3) / 2
DEADLINE = 135.0  # safety margin before 150s kill

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
VERTS = np.array([V0, V1, V2])

# Precompute all C(11,3) = 165 triplets
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)))
_T_I = _TRIPLES[:, 0]
_T_J = _TRIPLES[:, 1]
_T_K = _TRIPLES[:, 2]


def _min_area(points):
    """Minimum triangle area among all C(N,3) triples."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return float(areas.min())


def _sym_params_to_points(s):
    """
    Convert symmetric parameter vector to 11 points.
    Symmetry: reflection about x=0.5 (vertical axis).
    s = [x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, xc,yc, x6,y6]
    5 left points, 1 center point, 1 top point -> 11 total
    Right points are reflections of left points.
    """
    pts = np.zeros((N, 2))
    for i in range(5):
        pts[i, 0] = s[2 * i]
        pts[i, 1] = s[2 * i + 1]
    # Center point
    pts[5, 0] = s[10]
    pts[5, 1] = s[11]
    # Top point (on axis)
    pts[6, 0] = s[12]
    pts[6, 1] = s[13]
    # Right reflections of left points (indices 0-4 -> 7-11, but we only have 11)
    # We have 5 left + 1 center + 1 top = 7, need 4 more on right
    # Actually: 5 left, reflect 4 of them to right, 1 center, 1 top = 11
    # Let's redo: 4 left, 1 center, 1 top, 4 right = 10... no
    # 5 left, 5 right (reflections), 1 center = 11
    # But top should be on axis. Let me restructure:
    # 4 left, 4 right (reflections), 3 on axis (center-ish) = 11
    # Simpler: 5 left, 5 right reflections, 1 on axis
    for i in range(5):
        pts[7 + i, 0] = 1.0 - s[2 * i]
        pts[7 + i, 1] = s[2 * i + 1]
    # pts[6] is the on-axis point (index 6)
    # Wait, that's 5+5+1=11 but indices 0-4, 7-11, and 6. Good.
    return pts


def _sym_params_to_points_v2(s):
    """
    4 left, 4 right (reflections), 3 on axis = 11 points.
    s = [x1,y1, x2,y2, x3,y3, x4,y4, xa, ya, xb, yb, xc, yc]  (12 params)
    """
    pts = np.zeros((N, 2))
    for i in range(4):
        pts[i, 0] = s[2 * i]
        pts[i, 1] = s[2 * i + 1]
    # 3 on-axis points (indices 4, 5, 6)
    pts[4, 0] = s[8]
    pts[4, 1] = s[9]
    pts[5, 0] = s[10]
    pts[5, 1] = s[11]
    pts[6, 0] = s[12]
    pts[6, 1] = s[13]
    # 4 right reflections (indices 7-10)
    for i in range(4):
        pts[7 + i, 0] = 1.0 - s[2 * i]
        pts[7 + i, 1] = s[2 * i + 1]
    return pts


def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the equilateral triangle."""
    x, y = p
    if y < -eps:
        return False
    if y > H + eps:
        return False
    # Left edge: y <= sqrt(3) * x
    if y > np.sqrt(3) * x + eps:
        return False
    # Right edge: y <= sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1.0 - x) + eps:
        return False
    return True


def _all_inside(points, eps=1e-9):
    return all(_in_triangle(p, eps) for p in points)


def _objective_neg_min_area(params):
    """Objective: negative of min area (to minimize)."""
    pts = _sym_params_to_points_v2(params)
    if not _all_inside(pts, eps=1e-6):
        return 1e10
    return -_min_area(pts)


def _local_optimize(params0, time_budget):
    """Local optimization from initial params using L-BFGS-B with constraints."""
    start_time = time.time()
    best_params = params0.copy()
    best_val = _objective_neg_min_area(params0)
    best_area = -best_val if best_val < 1e9 else 0.0

    # Bounds: x in [0,1], y in [0,H]
    lb = np.zeros(14)
    ub = np.zeros(14)
    ub[::2] = 1.0  # x coords
    ub[1::2] = H   # y coords
    # On-axis points: x in [0,1]
    # Left points: x in [0, 0.5]
    ub[0:8:2] = 0.5
    # Right points derived, no direct bounds
    lb[8:14:2] = 0.0
    ub[8:14:2] = 1.0

    bounds = list(zip(lb, ub))

    for method in ['L-BFGS-B', 'Nelder-Mead']:
        if time.time() - start_time > time_budget:
            break
        try:
            res = minimize(_objective_neg_min_area, best_params, method=method,
                           bounds=bounds,
                           options={'maxiter': 500, 'maxfev': 5000,
                                    'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_val and res.fun < 1e9:
                best_val = res.fun
                best_params = res.x.copy()
                best_area = -best_val
        except Exception:
            pass

    return best_params, best_area, best_params


def _random_sym_params(rng, n_left=4):
    """Generate random symmetric parameters inside triangle."""
    s = np.zeros(14)
    for i in range(n_left):
        x = rng.uniform(0.02, 0.48)
        # y must satisfy y <= sqrt(3)*x and y <= sqrt(3)*(1-x) and y <= H
        y_max = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x), H)
        y = rng.uniform(0.01, max(0.02, y_max * 0.95))
        s[2 * i] = x
        s[2 * i + 1] = y
    # On-axis points: x in [0.2, 0.8], y constrained
    for j in range(3):
        x = rng.uniform(0.2, 0.8)
        y_max = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x), H)
        y = rng.uniform(0.01, max(0.02, y_max * 0.95))
        s[8 + 2 * j] = x
        s[9 + 2 * j] = y
    return s


def _grid_sym_params():
    """Generate structured grid of symmetric parameters."""
    params_list = []
    # Place points in a structured way
    xs_left = [0.15, 0.25, 0.35, 0.45]
    for x in xs_left:
        y_max = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x), H)
        for frac in [0.2, 0.4, 0.6, 0.8]:
            s = np.zeros(14)
            s[0] = x
            s[1] = y_max * frac
            # Fill others randomly-ish
            for i in range(1, 4):
                xi = 0.15 + 0.1 * i
                ym = min(np.sqrt(3) * xi, np.sqrt(3) * (1 - xi), H)
                s[2 * i] = xi
                s[2 * i + 1] = ym * 0.5
            for j in range(3):
                s[8 + 2 * j] = 0.5
                s[9 + 2 * j] = H * (0.3 + 0.2 * j)
            params_list.append(s)
    return params_list


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)

    best_area = 0.0
    best_params = None

    # Strategy 1: Try some structured initial points
    structured = _grid_sym_params()
    for s in structured[:10]:
        if time.time() - start_time > DEADLINE:
            break
        params, area, _ = _local_optimize(s, time_budget=2.0)
        if area > best_area:
            best_area = area
            best_params = params.copy()

    # Strategy 2: Random restarts with local optimization
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        s = _random_sym_params(rng)
        params, area, _ = _local_optimize(s, time_budget=3.0)
        if area > best_area:
            best_area = area
            best_params = params.copy()

    # Strategy 3: Differential evolution on symmetric params
    if time.time() - start_time < DEADLINE - 30:
        try:
            lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ub = np.array([0.5, H, 0.5, H, 0.5, H, 0.5, H, 1.0, H, 1.0, H, 1.0, H])
            time_left = DEADLINE - time.time() - 15
            if time_left > 20:
                de_res = differential_evolution(
                    _objective_neg_min_area,
                    bounds=list(zip(lb, ub)),
                    maxiter=100,
                    popsize=15,
                    tol=1e-8,
                    seed=42,
                    polish=True,
                    update_interval=1,
                    callback=lambda x, convergence: (time.time() - start_time > DEADLINE - 20)
                )
                if de_res.fun < 1e9 and -de_res.fun > best_area:
                    best_area = -de_res.fun
                    best_params = de_res.x.copy()
        except Exception:
            pass

    # Final polish: high-precision local optimization
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(_objective_neg_min_area, best_params, method='L-BFGS-B',
                           bounds=list(zip(np.zeros(14), np.array([0.5, H, 0.5, H, 0.5, H, 0.5, H, 1.0, H, 1.0, H, 1.0, H]))),
                           options={'maxiter': 1000, 'maxfev': 10000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < 1e9 and -res.fun > best_area:
                best_area = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: simple symmetric placement
        best_params = np.zeros(14)
        for i in range(4):
            best_params[2 * i] = 0.15 + 0.1 * i
            best_params[2 * i + 1] = 0.3
        for j in range(3):
            best_params[8 + 2 * j] = 0.5
            best_params[9 + 2 * j] = 0.3 + 0.2 * j

    pts = _sym_params_to_points_v2(best_params)
    return pts
