# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers, cx=0.0, cy=0.0, a_deg=0.0):
    """
    Minimum side length s of a regular hexagon centered at (cx,cy) with angle a_deg
    that contains all unit hexagons with given centers (angle 0, side 1).
    """
    a = np.deg2rad(a_deg)
    # Rotate centers into outer-hex frame so outer hex is axis-aligned at origin
    ca, sa = np.cos(a), np.sin(a)
    rx = ca * (centers[:, 0] - cx) - sa * (centers[:, 1] - cy)
    ry = sa * (centers[:, 0] - cx) + ca * (centers[:, 1] - cy)

    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    # All vertices in outer frame
    vx = rx[:, None] + base_cos[None, :]
    vy = ry[:, None] + base_sin[None, :]

    # Outward normals at 30,90,150,210,270,330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    nc = np.cos(normal_angles)
    ns = np.sin(normal_angles)

    # Projection of each vertex onto each normal
    proj = vx * nc[None, None, :] + vy * ns[None, None, :]  # (12,6,6)
    max_proj = np.max(proj)  # max over all vertices and normals
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _check_separation(centers, tol=1e-6):
    """SAT check: all pairs of unit hexagons separated."""
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    nc = np.cos(normal_angles)
    ns = np.sin(normal_angles)

    for i in range(n):
        vi = np.column_stack([centers[i, 0] + base_cos, centers[i, 1] + base_sin])
        for j in range(i + 1, n):
            vj = np.column_stack([centers[j, 0] + base_cos, centers[j, 1] + base_sin])
            separated = False
            for k in range(6):
                n_vec = np.array([nc[k], ns[k]])
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    separated = True
                    break
            if not separated:
                return False
    return True


def _separation_violation(centers):
    """Total squared overlap across all pairs (0 if separated)."""
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    nc = np.cos(normal_angles)
    ns = np.sin(normal_angles)

    total = 0.0
    for i in range(n):
        vi = np.column_stack([centers[i, 0] + base_cos, centers[i, 1] + base_sin])
        for j in range(i + 1, n):
            vj = np.column_stack([centers[j, 0] + base_cos, centers[j, 1] + base_sin])
            max_overlap = 0.0
            for k in range(6):
                n_vec = np.array([nc[k], ns[k]])
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                overlap = min(np.max(proj_i), np.max(proj_j)) - max(np.min(proj_i), np.min(proj_j))
                if overlap > max_overlap:
                    max_overlap = overlap
            if max_overlap > 0:
                total += max_overlap ** 2
    return total


def _construct_12_hex_cluster():
    """
    Construct 12 unit hexagons in a compact hexagonal cluster:
    - 1 center at origin
    - Ring 1: 6 hexagons at distance sqrt(3) at angles 0,60,...,300
    - Ring 2 (partial): 6 hexagons at distance 2*sqrt(3) at angles 30,90,...,330
    This is the natural hexagonal packing of 12 hexagons.
    """
    sqrt3 = np.sqrt(3.0)
    centers = np.zeros((12, 2))
    centers[0] = [0.0, 0.0]
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        centers[1 + k] = [sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)]
    # Ring 2: 6 hexagons at intermediate angles
    for k in range(6):
        ang = np.deg2rad(60.0 * k + 30.0)
        centers[7 + k] = [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]
    return centers


def _construct_12_hex_cluster_alt():
    """
    Alternative: 12 hexagons in a 3x4-ish rectangular arrangement optimized for hexagonal container.
    Using a hexagonal lattice with rows.
    """
    sqrt3 = np.sqrt(3.0)
    # Flat-top hexagons: horizontal spacing 1.5, vertical sqrt(3)
    # Arrange in 3 rows of 4
    centers = np.zeros((12, 2))
    idx = 0
    for row in range(3):
        y = row * sqrt3
        for col in range(4):
            x = col * 1.5 + (0.75 if row % 2 == 1 else 0.0)
            centers[idx] = [x, y]
            idx += 1
    # Center the cluster
    centers -= centers.mean(axis=0)
    return centers


def _construct_12_hex_cluster_v2():
    """
    Another arrangement: 2-3-3-2-2 pattern or similar compact shape.
    Actually, let's try a more hexagonal-symmetric arrangement.
    12 = 1 + 6 + 5 (partial ring 2)
    Or 12 = 1 + 6 + 6 (full ring 2) -- that's 13, too many.
    Let's do 1 + 6 + 5: center, ring 1 (6), ring 2 (5 out of 6)
    """
    sqrt3 = np.sqrt(3.0)
    centers = np.zeros((12, 2))
    centers[0] = [0.0, 0.0]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        centers[1 + k] = [sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)]
    # Ring 2: 5 hexagons (skip one to make 12 total)
    for k in range(5):
        ang = np.deg2rad(60.0 * k + 30.0)
        centers[7 + k] = [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]
    return centers


def _refine_3dof(centers_init):
    """
    Refine using 3 DOF: translate (tx, ty) and rotate (a_deg) the cluster,
    minimizing the containing hexagon side length.
    The cluster shape is fixed; we only optimize placement/orientation.
    """
    def obj(params):
        tx, ty, a_deg = params
        a = np.deg2rad(a_deg)
        ca, sa = np.cos(a), np.sin(a)
        # Rotate and translate
        rx = ca * centers_init[:, 0] - sa * centers_init[:, 1] + tx
        ry = sa * centers_init[:, 0] + ca * centers_init[:, 1] + ty
        # Containing hex centered at origin, angle 0
        return _min_containing_hex_side(np.column_stack([rx, ry]), 0.0, 0.0, 0.0)

    best_s = np.inf
    best_params = None

    # Try multiple starting orientations
    for a0 in range(0, 360, 15):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        x0 = np.array([0.0, 0.0, float(a0)])
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'maxfev': 50000,
                                    'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        except Exception:
            continue
        if res is not None and res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()

    if best_params is None:
        return centers_init, _min_containing_hex_side(centers_init)

    tx, ty, a_deg = best_params
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    rx = ca * centers_init[:, 0] - sa * centers_init[:, 1] + tx
    ry = sa * centers_init[:, 0] + ca * centers_init[:, 1] + ty
    refined = np.column_stack([rx, ry])
    return refined, best_s


def _full_24dof_refine(centers_init):
    """
    Full 24-DOF Nelder-Mead refinement as fallback/supplement.
    """
    def obj(params):
        c = params.reshape(12, 2)
        s = _min_containing_hex_side(c)
        v = _separation_violation(c)
        return s + 100.0 * v

    x0 = centers_init.flatten()
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'maxfev': 100000,
                                'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    except Exception:
        return centers_init, _min_containing_hex_side(centers_init)

    if res is not None:
        candidate = res.x.reshape(12, 2)
        if _check_separation(candidate):
            s_cand = _min_containing_hex_side(candidate)
            return candidate, s_cand
    return centers_init, _min_containing_hex_side(centers_init)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Generate multiple candidate constructions
    candidates = [
        _construct_12_hex_cluster(),
        _construct_12_hex_cluster_alt(),
        _construct_12_hex_cluster_v2(),
    ]

    best_centers = None
    best_s = np.inf
    best_outer_angle = 0.0

    rng = np.random.default_rng(42)

    for init_idx, init_centers in enumerate(candidates):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break

        # Check if initial is valid (separated)
        if not _check_separation(init_centers):
            # Try to fix with small perturbation
            continue

        s_init = _min_containing_hex_side(init_centers)
        if s_init < best_s:
            best_s = s_init
            best_centers = init_centers.copy()
            best_outer_angle = 0.0

        # 3-DOF refinement (translate + rotate)
        refined, s_ref = _refine_3dof(init_centers)
        if _check_separation(refined):
            s_check = _min_containing_hex_side(refined)
            if s_check < best_s:
                best_s = s_check
                best_centers = refined.copy()
                best_outer_angle = 0.0

        # Add random perturbations and re-refine
        for r in range(3):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            noise = rng.normal(0, 0.2, size=init_centers.shape)
            perturbed = init_centers + noise
            if _check_separation(perturbed):
                refined_p, s_p = _refine_3dof(perturbed)
                if _check_separation(refined_p):
                    s_cp = _min_containing_hex_side(refined_p)
                    if s_cp < best_s:
                        best_s = s_cp
                        best_centers = refined_p.copy()
                        best_outer_angle = 0.0

    # If we have a good solution, do a final 24-DOF polish
    if best_centers is not None:
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN:
            polished, s_pol = _full_24dof_refine(best_centers)
            if s_pol < best_s:
                best_s = s_pol
                best_centers = polished.copy()

    # Fallback if nothing valid found
    if best_centers is None:
        best_centers = _construct_12_hex_cluster()
        best_s = _min_containing_hex_side(best_centers)

    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
