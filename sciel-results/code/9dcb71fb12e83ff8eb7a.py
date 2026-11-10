# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_N = 13
_TRIS = np.array([(i, j, k) for i in range(_N) for j in range(i+1, _N) for k in range(j+1, _N)], dtype=np.int32)
_I_IDX = _TRIS[:, 0]
_J_IDX = _TRIS[:, 1]
_K_IDX = _TRIS[:, 2]


def _triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas


def _score(pts):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    if len(pts) < 3:
        return 0.0
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _soft_min_and_grad(pts, beta):
    """
    Compute soft-min of triangle areas and its gradient w.r.t. point coordinates.
    Returns (soft_min_value, grad_26).
    """
    areas = _triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_shifted = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_val + np.log(sum_exp)
    soft_min = -log_sum / beta
    
    weights = exp_shifted / sum_exp
    
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    grad = np.zeros(2 * _N)
    
    dx_i = signs * (y2 - y3) / 2.0
    dy_i = signs * (x3 - x2) / 2.0
    dx_j = signs * (y3 - y1) / 2.0
    dy_j = signs * (x3 - x1) / 2.0
    dx_k = signs * (y1 - y2) / 2.0
    dy_k = signs * (x2 - x1) / 2.0
    
    w = weights
    
    np.add.at(grad, _I_IDX, w * dx_i)
    np.add.at(grad, _I_IDX + _N, w * dy_i)
    np.add.at(grad, _J_IDX, w * dx_j)
    np.add.at(grad, _J_IDX + _N, w * dy_j)
    np.add.at(grad, _K_IDX, w * dx_k)
    np.add.at(grad, _K_IDX + _N, w * dy_k)
    
    return soft_min, grad


def _objective_and_grad(x, beta):
    """
    Objective: -soft_min / hull_area, and its gradient.
    """
    pts = np.column_stack([x[0:_N], x[_N:2*_N]])
    soft_min, grad_soft = _soft_min_and_grad(pts, beta)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10, np.zeros(2*_N)
    
    if hull_area <= 0 or not np.isfinite(soft_min) or not np.isfinite(hull_area):
        return 1e10, np.zeros(2*_N)
    
    obj = -soft_min / hull_area
    grad = -grad_soft / hull_area
    
    return obj, grad


def _regular_polygon(n=13):
    """Create regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _hexagonal_lattice_init():
    """
    Create a hexagonal lattice arrangement of 13 points.
    The hexagonal lattice has points at positions:
    - Center: (0, 0)
    - First ring: 6 points at distance 1, angles 0, 60, 120, 180, 240, 300
    - Second ring: 6 points at distance 2, angles 30, 90, 150, 210, 270, 330
    Total: 1 + 6 + 6 = 13 points
    """
    pts = np.zeros((13, 2))
    # Center
    pts[0] = [0, 0]
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        pts[1 + i] = [np.cos(angle), np.sin(angle)]
    # Second ring (6 points) - offset by 30 degrees
    for i in range(6):
        angle = np.pi/6 + i * np.pi / 3
        pts[7 + i] = [2 * np.cos(angle), 2 * np.sin(angle)]
    return pts


def _hexagonal_deformed_init():
    """
    Create a deformed hexagonal lattice where the basis vectors are non-orthogonal,
    allowing the convex hull to be a parallelogram or hexagon.
    """
    # Basis vectors for hexagonal lattice
    # Standard hex: v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
    # Deform: v1 = (1, 0.2), v2 = (0.4, 0.9)
    v1 = np.array([1.0, 0.2])
    v2 = np.array([0.4, 0.9])
    
    pts = np.zeros((13, 2))
    # Center
    pts[0] = [0, 0]
    # First ring: +/-v1, +/-v2, +/- (v1+v2)
    pts[1] = v1
    pts[2] = -v1
    pts[3] = v2
    pts[4] = -v2
    pts[5] = v1 + v2
    pts[6] = -(v1 + v2)
    # Second ring: 2*v1, 2*v2, 2*(v1+v2), 2*v1-v2, 2*v2-v1, v1-2*v2
    pts[7] = 2 * v1
    pts[8] = 2 * v2
    pts[9] = 2 * (v1 + v2)
    pts[10] = 2 * v1 - v2
    pts[11] = 2 * v2 - v1
    pts[12] = v1 - 2 * v2
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 590  # Safety margin from 630s

    n = 13
    best_pts = _regular_polygon(n)
    best_score = _score(best_pts)

    # Try multiple initializations with different structures
    initializations = []

    # 1. Regular polygon
    initializations.append(_regular_polygon(n))

    # 2. Hexagonal lattice (standard)
    initializations.append(_hexagonal_lattice_init())

    # 3. Deformed hexagonal lattice
    initializations.append(_hexagonal_deformed_init())

    # 4. Two concentric rings
    for inner_n, outer_n in [(5, 8), (6, 7), (4, 9)]:
        inner_angles = np.linspace(0, 2*np.pi, inner_n, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, outer_n, endpoint=False) + np.pi/outer_n
        inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initializations.append(np.vstack([inner_pts, outer_pts]))

    # 5. Grid-like arrangement
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([0.3 + 0.4*i, 0.3 + 0.4*j])
    grid_pts = np.array(grid_pts[:n])
    initializations.append(grid_pts)

    # 6. Slightly perturbed regular polygon
    for seed in range(5):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.1, (n, 2))
        initializations.append(perturbed)

    # 7. Perturbed hexagonal lattice
    for seed in range(3):
        rng2 = np.random.default_rng(100 + seed)
        hex_pts = _hexagonal_lattice_init()
        hex_pts += rng2.normal(0, 0.05, (n, 2))
        initializations.append(hex_pts)

    # 8. Perturbed deformed hex
    for seed in range(3):
        rng2 = np.random.default_rng(200 + seed)
        hex_pts = _hexagonal_deformed_init()
        hex_pts += rng2.normal(0, 0.05, (n, 2))
        initializations.append(hex_pts)

    # Optimize each initialization
    beta_schedule = [30, 80, 200, 500, 1500, 5000]

    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 40:
            break

        x0 = init_pts.flatten()
        current_x = x0.copy()
        
        for beta in beta_schedule:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 20:
                break

            # L-BFGS-B with analytical gradient
            try:
                result = minimize(
                    lambda x, b=beta: _objective_and_grad(x, b)[0],
                    current_x,
                    method='L-BFGS-B',
                    jac=lambda x, b=beta: _objective_and_grad(x, b)[1],
                    options={
                        'maxiter': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                if result.fun < _objective_and_grad(current_x, beta)[0] - 1e-10:
                    current_x = result.x
            except Exception:
                pass

            # Nelder-Mead polish
            try:
                nm_result = minimize(
                    lambda x, b=beta: _objective_and_grad(x, b)[0],
                    current_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-12,
                        'fatol': 1e-14
                    }
                )
                if nm_result.fun < _objective_and_grad(current_x, beta)[0] - 1e-10:
                    current_x = nm_result.x
            except Exception:
                pass

        # Evaluate final result
        final_pts = np.column_stack([current_x[0:13], current_x[13:26]])
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final high-precision polish with exact objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20 and best_score > 0:
        x_final = best_pts.flatten()

        def exact_obj(x):
            pts = np.column_stack([x[0:13], x[13:26]])
            s = _score(pts)
            return -s if s > 0 else 1e10

        try:
            nm_final = minimize(
                exact_obj,
                x_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16
                }
            )
            if nm_final.fun < -best_score:
                best_pts = np.column_stack([nm_final.x[0:13], nm_final.x[13:26]])
                best_score = -nm_final.fun
        except Exception:
            pass

    # Validate result
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(n)

    return best_pts


# EVOLVE-BLOCK-END
