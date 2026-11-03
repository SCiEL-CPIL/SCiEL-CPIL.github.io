"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area_vec(points):
    """Vectorized: min area of any triangle formed by the points."""
    n = len(points)
    # Build all triples (i, j, k) with i < j < k
    idx = np.array([[i, j, k] for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)])
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # cross product magnitude / 2
    dx12 = p2[:, 0] - p1[:, 0]
    dy12 = p2[:, 1] - p1[:, 1]
    dx13 = p3[:, 0] - p1[:, 0]
    dy13 = p3[:, 1] - p1[:, 1]
    cross = dx12 * dy13 - dy12 * dx13
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _in_triangle_vec(points, eps=1e-4):
    """Vectorized check: all points inside equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x = points[:, 0]
    y = points[:, 1]
    h = np.sqrt(3) / 2
    # y bounds
    ok = (y >= -eps) & (y <= h + eps) & (x >= -eps) & (x <= 1 + eps)
    # left edge: y <= sqrt(3)*x  for x <= 0.5
    left_mask = x < 0.5
    ok[left_mask] &= (y[left_mask] <= np.sqrt(3) * x[left_mask] + eps)
    # right edge: y <= sqrt(3)*(1-x) for x >= 0.5
    right_mask = ~left_mask
    ok[right_mask] &= (y[right_mask] <= np.sqrt(3) * (1 - x[right_mask]) + eps)
    return np.all(ok)


def _objective_flat(flat_params):
    """Objective: minimize negative min triangle area. Returns large penalty if out of bounds."""
    pts = np.empty((N, 2))
    u = flat_params[0::2]
    v = flat_params[1::2]
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = (np.sqrt(3) / 2) * v
    if not _in_triangle_vec(pts, eps=0.005):
        return 1.0  # penalty (we minimize, so large = bad)
    return -_min_triangle_area_vec(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin (evaluator kills at 150s)

    # Good initial configuration: symmetric distribution in barycentric coords
    # Barycentric (a, b, c) -> cartesian: x = a + 0.5*b, y = sqrt(3)/2 * b
    # where a = 1 - b - c (since a+b+c=1, point = a*(0,0) + b*(1,0) + c*(0.5, sqrt(3)/2))
    # Actually let's use: point = b*(1,0) + c*(0.5, sqrt(3)/2)
    # So x = b + 0.5*c, y = sqrt(3)/2 * c
    # In our flat params: u corresponds to b, v corresponds to c
    # x = u + 0.5*v, y = sqrt(3)/2 * v
    # Triangle constraint: u >= 0, v >= 0, u + v <= 1

    # Symmetric 11-point initial guess
    # Use known good configurations: corners + edge midpoints + interior
    barycentric_init = [
        [0.05, 0.05, 0.90],   # near corner (0,0)
        [0.90, 0.05, 0.05],   # near corner (1,0)
        [0.05, 0.90, 0.05],   # near corner (0.5, h)
        [0.50, 0.05, 0.45],   # bottom edge left-ish
        [0.05, 0.50, 0.45],   # right edge
        [0.45, 0.45, 0.10],   # left edge
        [0.33, 0.33, 0.34],   # center
        [0.20, 0.55, 0.25],
        [0.55, 0.20, 0.25],
        [0.25, 0.25, 0.50],
        [0.50, 0.25, 0.25],
    ]
    # Convert barycentric (a, b, c) to flat (u, v) where u=b, v=c
    initial_flat = []
    for a, b, c in barycentric_init:
        initial_flat.extend([b, c])
    initial_flat = np.array(initial_flat)

    # Bounds: u in [0,1], v in [0,1] (penalty handles triangle constraint)
    bounds = [(0, 1), (0, 1)] * N

    best_points = None
    best_score = -1.0

    # Phase 1: Differential Evolution
    de_start = time.time()
    try:
        result = differential_evolution(
            _objective_flat,
            bounds,
            x0=initial_flat,
            maxiter=300,
            popsize=15,
            tol=1e-7,
            mutation=(0.4, 1.0),
            recombination=0.8,
            polish=False,  # we'll do our own polish
            seed=42,
            init='sobol',
            callback=None
        )
        if result.fun < 0:  # valid solution
            best_flat = result.x
            best_score = -result.fun
            pts = np.empty((N, 2))
            u = best_flat[0::2]
            v = best_flat[1::2]
            pts[:, 0] = u + 0.5 * v
            pts[:, 1] = (np.sqrt(3) / 2) * v
            best_points = pts
    except Exception:
        pass

    elapsed = time.time() - start_time
    if best_points is None:
        # Fallback: use initial
        pts = np.empty((N, 2))
        u = initial_flat[0::2]
        v = initial_flat[1::2]
        pts[:, 0] = u + 0.5 * v
        pts[:, 1] = (np.sqrt(3) / 2) * v
        best_points = pts
        best_score = _min_triangle_area_vec(pts)

    # Phase 2: Local refinement with Nelder-Mead on the best DE result
    if time.time() < deadline:
        try:
            best_flat_arr = np.array([best_points[i, 0] - 0.5 * best_points[i, 1] for i in range(N)] +
                                     [best_points[i, 1] * 2 / np.sqrt(3) for i in range(N)])
            # Actually let's re-extract from the stored flat
            if result is not None and result.fun < 0:
                best_flat_arr = result.x.copy()
            else:
                best_flat_arr = initial_flat.copy()

            res_nm = minimize(
                _objective_flat,
                best_flat_arr,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < best_score * -1:  # better (more negative objective = larger area)
                best_score = -res_nm.fun
                pts = np.empty((N, 2))
                u = res_nm.x[0::2]
                v = res_nm.x[1::2]
                pts[:, 0] = u + 0.5 * v
                pts[:, 1] = (np.sqrt(3) / 2) * v
                if _in_triangle_vec(pts, eps=1e-4):
                    best_points = pts
        except Exception:
            pass

    # Phase 3: If time remains, try a few random restarts with short DE
    if time.time() < deadline - 30 and best_points is not None:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            try:
                # Random perturbation of current best
                perturbation = rng.randn(2 * N) * 0.05
                x0_restart = best_flat_arr + perturbation
                x0_restart = np.clip(x0_restart, 0, 1)

                res_r = differential_evolution(
                    _objective_flat,
                    bounds,
                    x0=x0_restart,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    polish=False,
                    seed=42 + restart,
                )
                if res_r.fun < 0 and -res_r.fun > best_score:
                    best_score = -res_r.fun
                    pts = np.empty((N, 2))
                    u = res_r.x[0::2]
                    v = res_r.x[1::2]
                    pts[:, 0] = u + 0.5 * v
                    pts[:, 1] = (np.sqrt(3) / 2) * v
                    if _in_triangle_vec(pts, eps=1e-4):
                        best_points = pts
                        best_flat_arr = res_r.x.copy()
            except Exception:
                break

    # Final: ensure valid
    if best_points is None or not _in_triangle_vec(best_points, eps=1e-3):
        # Absolute fallback
        pts = np.empty((N, 2))
        u = initial_flat[0::2]
        v = initial_flat[1::2]
        pts[:, 0] = u + 0.5 * v
        pts[:, 1] = (np.sqrt(3) / 2) * v
        best_points = pts

    return best_points
# EVOLVE-BLOCK-END
