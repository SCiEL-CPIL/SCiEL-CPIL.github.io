import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 45.0


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _precompute_base():
    """Precompute base vertices and normals for unit hexagon at angle 0."""
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    base_dots = base_verts @ normals.T
    base_min = base_dots.min(axis=0)
    base_max = base_dots.max(axis=0)
    return base_verts, normals, base_min, base_max


_BASE_VERTS, _NORMALS, _BASE_MIN, _BASE_MAX = _precompute_base()


def _check_separation_fast(positions):
    """Check if all pairs of unit hexagons at given positions (same angle 0) are separated."""
    n = len(positions)
    center_dots = positions @ _NORMALS.T
    for i in range(n):
        for j in range(i + 1, n):
            proj_i_min = center_dots[i] + _BASE_MIN
            proj_i_max = center_dots[i] + _BASE_MAX
            proj_j_min = center_dots[j] + _BASE_MIN
            proj_j_max = center_dots[j] + _BASE_MAX
            sep1 = proj_i_max < proj_j_min - 1e-6
            sep2 = proj_j_max < proj_i_min - 1e-6
            if np.any(sep1) or np.any(sep2):
                continue
            else:
                return False
    return True


def _separation_violation_fast(positions):
    """Compute total separation violation for same-angle hexagons."""
    n = len(positions)
    center_dots = positions @ _NORMALS.T
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            proj_i_min = center_dots[i] + _BASE_MIN
            proj_i_max = center_dots[i] + _BASE_MAX
            proj_j_min = center_dots[j] + _BASE_MIN
            proj_j_max = center_dots[j] + _BASE_MAX
            overlap = np.minimum(proj_i_max, proj_j_max) - np.maximum(proj_i_min, proj_j_min)
            max_overlap = np.max(overlap)
            if max_overlap > 0:
                total += max_overlap ** 2
    return total


def _min_containing_hex_side_fast(centers, angle_deg=0.0):
    """Compute minimum side length of regular hexagon (centered at origin) containing all unit hexagons."""
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    center_dots = centers @ normals.T
    base_dots = _BASE_VERTS @ normals.T
    all_proj = center_dots[:, :, np.newaxis] + base_dots[np.newaxis, :, :]
    max_proj = np.max(all_proj)
    return (2.0 / np.sqrt(3.0)) * max_proj


def _check_validity(inner, outer, s):
    """Full validation per task contract."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s too small"
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    for row in inner:
        hv = _hex_vertices(row[0], row[1], 1.0, row[2])
        for i in range(6):
            if abs(np.linalg.norm(hv[(i + 1) % 6] - hv[i]) - 1.0) > 1e-8:
                return False, "edge length"
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            v1, v2 = inner_verts[i], inner_verts[j]
            sep = False
            for vi in [v1, v2]:
                for k in range(6):
                    edge = vi[(k + 1) % 6] - vi[k]
                    n = np.array([-edge[1], edge[0]])
                    nn = np.linalg.norm(n)
                    if nn < 1e-12:
                        continue
                    n /= nn
                    p1 = v1 @ n
                    p2 = v2 @ n
                    if np.min(p2) - np.max(p1) > 1e-6 or np.min(p1) - np.max(p2) > 1e-6:
                        sep = True
                        break
                if sep:
                    break
            if not sep:
                return False, f"overlap {i},{j}"
    for row in inner:
        hv = _hex_vertices(row[0], row[1], 1.0, row[2])
        for v in hv:
            for k in range(6):
                p1, p2 = ov[k], ov[(k + 1) % 6]
                edge = p2 - p1
                cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                if cross < -1e-6:
                    return False, "vertex outside"
    return True, ""


def _make_hex_cluster():
    """Compact hexagonal lattice of 12 unit hexagons (flat-top, same angle)."""
    sqrt3 = np.sqrt(3)
    positions = np.array([
        [0.0, 0.0],
        [1.5, sqrt3 / 2],
        [1.5, -sqrt3 / 2],
        [0.0, sqrt3],
        [0.0, -sqrt3],
        [-1.5, sqrt3 / 2],
        [-1.5, -sqrt3 / 2],
        [3.0, 0.0],
        [2.25, 3 * sqrt3 / 2],
        [0.75, 2 * sqrt3],
        [-1.5, 3 * sqrt3 / 2],
        [-3.0, 0.0],
    ])
    return positions


def _make_circular(r=2.0):
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _make_spread():
    sqrt3 = np.sqrt(3)
    positions = np.array([
        [0.0, 0.0],
        [-2.0, 0.0],
        [2.0, 0.0],
        [-1.0, sqrt3],
        [1.0, sqrt3],
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        [-3.0, sqrt3],
        [3.0, sqrt3],
        [-3.0, -sqrt3],
        [3.0, -sqrt3],
        [0.0, -2 * sqrt3],
    ])
    return positions


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s).
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        centers = _make_hex_cluster()
        inner = np.zeros((12, 3))
        inner[:, :2] = centers
        outer = np.array([0.0, 0.0, 0.0])
        s = _min_containing_hex_side_fast(centers, 0.0) + 0.01
        return inner, outer, s

    rng = np.random.default_rng(42)
    best_s = np.inf
    best_inner = None
    best_outer = None

    # Initial configurations
    inits = []
    base = _make_hex_cluster()
    inits.append(base)
    inits.append(base + rng.normal(0, 0.3, size=(12, 2)))
    inits.append(base + rng.normal(0, 0.5, size=(12, 2)))
    inits.append(_make_circular(2.0))
    inits.append(_make_circular(2.5))
    inits.append(_make_circular(3.0))
    inits.append(_make_spread())
    inits.append(_make_spread() + rng.normal(0, 0.3, size=(12, 2)))
    inits.append(rng.normal(0, 2.0, size=(12, 2)))
    inits.append(rng.normal(0, 2.5, size=(12, 2)))

    def _objective(params):
        """params: 25 values = 12*(x,y) + angle_deg."""
        positions = params[:24].reshape(12, 2)
        angle_deg = params[24]
        if not _check_separation_fast(positions):
            v = _separation_violation_fast(positions)
            return 100.0 + 10.0 * v
        return _min_containing_hex_side_fast(positions, angle_deg)

    def _try_optimize(init_pos, maxiter=3000):
        nonlocal best_s, best_inner, best_outer
        init_params = np.zeros(25)
        init_params[:24] = init_pos.flatten()
        init_params[24] = 0.0
        try:
            res = minimize(_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
        except Exception:
            return
        positions = res.x[:24].reshape(12, 2)
        angle = res.x[24]
        s_val = _min_containing_hex_side_fast(positions, angle) + 0.001
        inner = np.zeros((12, 3))
        inner[:, :2] = positions
        inner[:, 2] = 0.0
        outer = np.array([0.0, 0.0, angle])
        valid, _ = _check_validity(inner, outer, s_val)
        if valid and s_val < best_s:
            best_s = s_val
            best_inner = inner.copy()
            best_outer = outer.copy()

    for i, init_pos in enumerate(inits):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        _try_optimize(init_pos, maxiter=2000)

    # Polish
    if best_inner is not None and time.time() < _DEADLINE - _SAFETY_MARGIN:
        init_params = np.zeros(25)
        init_params[:24] = best_inner[:, :2].flatten()
        init_params[24] = best_outer[2]
        try:
            res = minimize(_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            positions = res.x[:24].reshape(12, 2)
            angle = res.x[24]
            s_val = _min_containing_hex_side_fast(positions, angle) + 0.0005
            inner = np.zeros((12, 3))
            inner[:, :2] = positions
            inner[:, 2] = 0.0
            outer = np.array([0.0, 0.0, angle])
            valid, _ = _check_validity(inner, outer, s_val)
            if valid and s_val < best_s:
                best_s = s_val
                best_inner = inner.copy()
                best_outer = outer.copy()
        except Exception:
            pass

    # Fallback
    if best_inner is None:
        centers = _make_hex_cluster()
        best_inner = np.zeros((12, 3))
        best_inner[:, :2] = centers
        best_outer = np.array([0.0, 0.0, 0.0])
        best_s = _min_containing_hex_side_fast(centers, 0.0) + 0.01

    valid, _ = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        centers = _make_hex_cluster()
        best_inner = np.zeros((12, 3))
        best_inner[:, :2] = centers
        best_outer = np.array([0.0, 0.0, 0.0])
        best_s = _min_containing_hex_side_fast(centers, 0.0) + 0.01

    return best_inner, best_outer, best_s
