"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute all triplet indices for speed
_TRIPLETS = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
_N_TRIPLETS = len(_TRIPLETS)  # 165


def _all_areas_bary(points_bary):
    """Compute all 165 triangle areas (as ratios) for N points in barycentric coords."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    areas = np.empty(_N_TRIPLETS)
    for idx, (i, j, k) in enumerate(_TRIPLETS):
        det = (u[i] * (v[j] * w[k] - w[j] * v[k])
               - v[i] * (u[j] * w[k] - w[j] * u[k])
               + w[i] * (u[j] * v[k] - v[j] * u[k]))
        areas[idx] = abs(det)
    return areas


def _min_area_bary(points_bary):
    """Calculate the minimum area ratio of triangles formed by N points."""
    return np.min(_all_areas_bary(points_bary))


def _clamp_to_simplex(points):
    """Clamp points to be inside the simplex u>=0, v>=0, u+v<=1."""
    u = np.clip(points[:, 0], 0, 1)
    v = np.clip(points[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1
    if np.any(mask):
        factor = 1.0 / sum_uv[mask]
        u[mask] *= factor
        v[mask] *= factor
    return np.column_stack((u, v))


def _bary_to_cartesian(points_bary):
    """Convert barycentric (u,v) to Cartesian in equilateral triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    u = np.clip(points_bary[:, 0], 0, 1)
    v = np.clip(points_bary[:, 1], 0, 1)
    w = 1.0 - u - v
    # Normalize in case of tiny negative w
    s = u + v + w
    s[s <= 0] = 1.0
    u /= s
    v /= s
    w /= s
    return u * A + v * B + w * C


def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric (u,v) in equilateral triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    # Solve: p = u*A + v*B + w*C, u+v+w=1
    # p_x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + v - 0.5*u - 0.5*v = 0.5 + 0.5*v - 0.5*u
    # p_y = (sqrt(3)/2)*w = (sqrt(3)/2)*(1-u-v)
    # From p_y: 1-u-v = 2*p_y/sqrt(3) => u+v = 1 - 2*p_y/sqrt(3)
    # From p_x: 0.5 + 0.5*v - 0.5*u = p_x => v - u = 2*p_x - 1 => v = u + 2*p_x - 1
    # So: u + u + 2*p_x - 1 = 1 - 2*p_y/sqrt(3) => 2u = 2 - 2*p_x - 2*p_y/sqrt(3)
    # u = 1 - p_x - p_y/sqrt(3)
    # v = u + 2*p_x - 1 = 2*p_x - p_y/sqrt(3)
    u = 1.0 - pts[:, 0] - pts[:, 1] / np.sqrt(3)
    v = 2.0 * pts[:, 0] - pts[:, 1] / np.sqrt(3)
    return np.column_stack((u, v))


def _min_area_cartesian(pts):
    """Compute minimum triangle area for points in Cartesian coords (equilateral triangle)."""
    # Area of triangle = 0.5 * |cross product|
    # But we want ratio to total area. Total area of equilateral triangle with side 1 is sqrt(3)/4.
    # Actually, the barycentric determinant gives area ratio directly.
    # Let's just convert to barycentric and use _min_area_bary for consistency.
    bary = _cartesian_to_bary(pts)
    bary = _clamp_to_simplex(bary)
    return _min_area_bary(bary)


def _objective_bary(params, penalty_weight=100.0):
    """Objective for L-BFGS-B in barycentric space: minimize -min_area + penalty."""
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v

    penalty = 0.0
    if np.any(w < 0):
        violations = -w[w < 0]
        penalty = penalty_weight * np.sum(violations ** 2)
    if np.any(u < 0):
        penalty += penalty_weight * np.sum(u[u < 0] ** 2)
    if np.any(v < 0):
        penalty += penalty_weight * np.sum(v[v < 0] ** 2)

    if penalty > 1e-10:
        return penalty - 1e-6

    return -_min_area_bary(uv) + penalty


def _objective_cartesian(params, penalty_weight=100.0):
    """Objective for Nelder-Mead in Cartesian space."""
    pts = params.reshape(N, 2)
    # Check if all points are inside the equilateral triangle
    # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For each point, check it's on the correct side of each edge
    # Edge 1: from (0,0) to (1,0): y >= 0
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2): line is y = -sqrt(3)*(x-1) => sqrt(3)*x + y - sqrt(3) <= 0
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0): line is y = sqrt(3)*x => sqrt(3)*x - y >= 0

    penalty = 0.0
    # Check edge 1: y >= 0
    below = pts[:, 1] < 0
    if np.any(below):
        penalty += penalty_weight * np.sum(pts[below, 1] ** 2)

    # Check edge 2: sqrt(3)*x + y - sqrt(3) <= 0
    val2 = np.sqrt(3) * pts[:, 0] + pts[:, 1] - np.sqrt(3)
    above = val2 > 0
    if np.any(above):
        penalty += penalty_weight * np.sum(val2[above] ** 2)

    # Check edge 3: sqrt(3)*x - y >= 0
    val3 = np.sqrt(3) * pts[:, 0] - pts[:, 1]
    below3 = val3 < 0
    if np.any(below3):
        penalty += penalty_weight * np.sum(val3[below3] ** 2)

    if penalty > 1e-10:
        return penalty - 1e-6

    return -_min_area_cartesian(pts)


def _get_good_initializations():
    """Generate a set of good starting points in barycentric coordinates."""
    inits = []
    rng = np.random.RandomState(42)

    # 1. Structured grid: 10-point grid + center (from inspiration 1)
    grid_10 = [
        (1.0, 0.0), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (0.0, 1.0),
        (2 / 3, 0.0), (1 / 3, 1 / 3), (0.0, 2 / 3),
        (1 / 3, 0.0), (0.0, 1 / 3),
        (0.0, 0.0)
    ]
    init1 = np.array(grid_10 + [(0.4, 0.4)])
    inits.append(init1)

    # 2. Vertices, edge midpoints, and interior points
    structured = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),  # vertices
        (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),  # edge midpoints
        (1 / 3, 1 / 3), (2 / 3, 1 / 6), (1 / 6, 2 / 3),  # interior
        (0.25, 0.25), (0.75, 0.125)
    ]
    init2 = np.array(structured[:N])
    inits.append(init2)

    # 3. Perturbed versions of structured initializations
    for base in [init1, init2]:
        for _ in range(20):
            noise = rng.uniform(-0.06, 0.06, size=(N, 2))
            perturbed = _clamp_to_simplex(base + noise)
            inits.append(perturbed)

    # 4. Random Dirichlet samples
    for _ in range(20):
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2])

    # 5. Symmetric hexagon-like pattern
    center = (1 / 3, 1 / 3)
    r = 0.25
    hex_pts = []
    for angle in range(0, 360, 60):
        rad = np.radians(angle)
        u = center[0] + r * np.cos(rad) * 0.5
        v = center[1] + r * (np.sin(rad) * 0.5 - np.cos(rad) * 0.25)
        hex_pts.append((u, v))
    interior = [(1 / 3, 1 / 3)]
    for angle in range(30, 360, 72):
        rad = np.radians(angle)
        u = center[0] + 0.1 * np.cos(rad) * 0.5
        v = center[1] + 0.1 * (np.sin(rad) * 0.5 - np.cos(rad) * 0.25)
        interior.append((u, v))
    init_hex = np.array(hex_pts[:6] + interior[:5])
    inits.append(init_hex)
    for _ in range(5):
        noise = rng.uniform(-0.05, 0.05, size=(N, 2))
        perturbed = _clamp_to_simplex(init_hex + noise)
        inits.append(perturbed)

    return inits


def _refine_bary(points_bary, deadline):
    """Refine points in barycentric space using L-BFGS-B."""
    current = points_bary.copy()
    bounds = [(0, 1)] * (2 * N)
    try:
        res = minimize(
            _objective_bary,
            current.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        uv_new = res.x.reshape(N, 2)
        w_new = 1 - uv_new[:, 0] - uv_new[:, 1]
        if np.all(w_new >= -1e-6) and np.all(uv_new >= -1e-6):
            uv_new = _clamp_to_simplex(uv_new)
            score_new = _min_area_bary(uv_new)
            score_old = _min_area_bary(current)
            if score_new > score_old:
                current = uv_new
    except Exception:
        pass
    return current


def _refine_cartesian(points_bary, deadline):
    """Refine points in Cartesian space using Nelder-Mead."""
    pts = _bary_to_cartesian(points_bary)
    current = pts.copy()
    try:
        res = minimize(
            _objective_cartesian,
            current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        pts_new = res.x.reshape(N, 2)
        # Convert back to barycentric and check validity
        bary_new = _cartesian_to_bary(pts_new)
        bary_new = _clamp_to_simplex(bary_new)
        score_new = _min_area_bary(bary_new)
        score_old = _min_area_bary(points_bary)
        if score_new > score_old:
            return bary_new
    except Exception:
        pass
    return points_bary


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_score = -float('inf')
    best_points_bary = None

    inits = _get_good_initializations()

    # Score and sort initializations
    init_scores = []
    for init in inits:
        score = _min_area_bary(init)
        init_scores.append((score, init))
    init_scores.sort(key=lambda x: x[0], reverse=True)

    # Process top initializations with multi-stage refinement
    for score, init in init_scores:
        if time.time() > deadline - 10:
            break

        current = init.copy()
        current_score = score

        # Stage 1: Barycentric L-BFGS-B refinement
        if time.time() < deadline - 15:
            current = _refine_bary(current, deadline)
            current_score = _min_area_bary(current)

        # Stage 2: Cartesian Nelder-Mead refinement
        if time.time() < deadline - 15:
            current = _refine_cartesian(current, deadline)
            current_score = _min_area_bary(current)

        # Stage 3: Another barycentric refinement after Cartesian
        if time.time() < deadline - 10:
            current = _refine_bary(current, deadline)
            current_score = _min_area_bary(current)

        if current_score > best_score:
            best_score = current_score
            best_points_bary = current.copy()

    if best_points_bary is None:
        best_points_bary = inits[0]

    # Convert barycentric to Cartesian
    pts = _bary_to_cartesian(best_points_bary)

    return pts

# EVOLVE-BLOCK-END
