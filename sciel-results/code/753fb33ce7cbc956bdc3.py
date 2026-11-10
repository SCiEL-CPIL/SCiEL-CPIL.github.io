# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_HEX_EDGE_ANGLES = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0


def _hex_vertices(cx, cy, d, angle_deg):
    a = np.deg2rad(angle_deg)
    return np.column_stack([
        cx + d * np.cos(a + _HEX_EDGE_ANGLES),
        cy + d * np.sin(a + _HEX_EDGE_ANGLES)
    ])


def _min_enclosing_hex_side(centers):
    """
    Find minimum side length s and angle a such that all centers are inside
    a regular hexagon of side s centered at origin.
    Uses scipy minimize on (s, angle).
    """
    def constraint(s, angle_deg):
        a = np.deg2rad(angle_deg)
        # For a regular hexagon centered at origin with side s and angle a,
        # the distance from center to edge in direction theta is:
        # d(theta) = s * cos(theta - a) / (1 - 0.5*cos(2*(theta-a)))  ... complex
        # Easier: the hexagon is intersection of 6 half-planes.
        # Edge i: normal direction = a + 60*i (for flat-top) or a + 30 + 60*i (for pointy-top)
        # Actually for a regular hexagon with vertices at angle a + 60*i,
        # the edges are perpendicular to directions a + 30 + 60*i
        # The distance from center to each edge is s * cos(30 deg) = s * sqrt(3)/2
        # Wait, that's the apothem. For a regular hexagon of side s, apothem = s * sqrt(3)/2.
        # The hexagon is: for each of 6 directions n_i (perpendicular to edges),
        # dot(center - 0, n_i) <= apothem
        # n_i directions: a + 30 deg + 60*i, i=0..5
        # apothem = s * sqrt(3) / 2
        n_dirs = a + np.pi/6 + np.arange(6) * np.pi/3
        proj = centers @ n_dirs.T  # (N, 6)
        max_proj = proj.max(axis=0)  # (6,)
        # Need max_proj <= s * sqrt(3)/2 for all edges
        # So s >= 2 * max_proj / sqrt(3)
        return 2 * max_proj / np.sqrt(3)

    def objective(params):
        s, angle_deg = params
        s = max(s, 1.0)
        required = constraint(s, angle_deg)
        # We want to minimize s, but the hexagon must contain all centers.
        # So s must be >= required. Minimize s subject to s >= required.
        # Use penalty: if s < required, penalty = (required - s)^2
        if s < required:
            return (required - s) ** 2 + s
        return s

    best_s = np.inf
    best_angle = 0
    for init_angle in np.linspace(0, 60, 13):
        res = minimize(objective, [5.0, init_angle], method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500})
        s_val = max(res.x[0], 1.0)
        a_val = res.x[1]
        required = constraint(s_val, a_val)
        s_val = max(s_val, required)
        if s_val < best_s:
            best_s = s_val
            best_angle = a_val

    return best_s, best_angle


def _hex_edge_normals(cx, cy, d, angle_deg):
    """Return the 6 outward unit normals of a hexagon."""
    a = np.deg2rad(angle_deg)
    # Normals point in directions a + 30 + 60*i
    n_dirs = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([np.cos(n_dirs), np.sin(n_dirs)])


def _check_separation(inner_data):
    """
    Check that all pairs of inner hexagons are separated.
    Returns minimum gap (positive = separated, negative = overlapping).
    """
    n = len(inner_data)
    min_gap = np.inf
    for i in range(n):
        cx1, cy1, a1 = inner_data[i]
        verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
        normals1 = _hex_edge_normals(cx1, cy1, 1.0, a1)
        for j in range(i+1, n):
            cx2, cy2, a2 = inner_data[j]
            verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
            normals2 = _hex_edge_normals(cx2, cy2, 1.0, a2)
            # SAT: check all normals from both hexagons
            for normals, verts_other in [(normals1, verts2), (normals2, verts1)]:
                for k in range(6):
                    n = normals[k]
                    # Project both hexagons onto this normal
                    p1 = verts1 @ n
                    p2 = verts_other @ n
                    gap = min(p2) - max(p1)
                    if gap < min_gap:
                        min_gap = gap
    return min_gap


def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside or on the outer hexagon (CCW vertices)."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True


def _validate(inner_data, outer_data, s):
    """Full validation. Returns (valid, min_separation, max_violation)."""
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False, -np.inf, 1.0

    # Check finite
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False, -np.inf, 1.0

    # Shift so outer center is at origin
    outer_cx, outer_cy, outer_angle = outer_data
    inner_shifted = inner_data.copy()
    inner_shifted[:, 0] -= outer_cx
    inner_shifted[:, 1] -= outer_cy

    # Check unit edges
    for i in range(12):
        cx, cy, a = inner_shifted[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j+1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False, -np.inf, 1.0

    # Check separation
    min_sep = _check_separation(inner_shifted)
    if min_sep < 1e-6:
        return False, min_sep, 0.0

    # Check all inner vertices inside outer
    outer_verts = _hex_vertices(outer_cx, outer_cy, s, outer_angle)
    max_viol = 0.0
    for i in range(12):
        cx, cy, a = inner_shifted[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < max_viol:
                    max_viol = cross
    if max_viol < -1e-6:
        return False, min_sep, -max_viol

    return True, min_sep, -max_viol


def _honeycomb_initial():
    """Create a honeycomb-like arrangement of 12 hexagons."""
    # Honeycomb: 1 center + 6 around + 5 more
    # Better: use a known tight packing
    dx = 2.5  # horizontal spacing for flat-top hexagons
    dy = 2.17  # approx sqrt(3)*1.25
    positions = [
        [0, 0],
        [-2.5, 0], [2.5, 0],
        [-1.25, 2.17], [1.25, 2.17],
        [-1.25, -2.17], [1.25, -2.17],
        [-3.75, 2.17], [3.75, 2.17],
        [-3.75, -2.17], [3.75, -2.17],
        [0, -4.34],
    ]
    return np.array(positions)


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620  # safety margin

    # Initial arrangement
    centers = _honeycomb_initial()
    angles = np.zeros(12)

    best_inner = None
    best_outer = None
    best_s = np.inf
    best_valid = False

    def _compute_s_and_angle(centers_local, angles_local):
        """Compute minimal enclosing hexagon for given centers and try to optimize angle."""
        # Try several angle initializations
        best_s_local = np.inf
        best_a_local = 0
        for init_a in np.linspace(0, 30, 7):
            # For a given outer angle, find min s
            # s = max over all vertices of max over all outer edge normals of (dot(v, n))
            # where n are the 6 normals of the outer hexagon
            a_rad = np.deg2rad(init_a)
            n_dirs = a_rad + np.pi/6 + np.arange(6) * np.pi/3
            max_s = 0
            for i in range(12):
                cx, cy = centers_local[i]
                a_i = np.deg2rad(angles_local[i])
                verts = np.column_stack([
                    cx + np.cos(a_i + _HEX_EDGE_ANGLES),
                    cy + np.sin(a_i + _HEX_EDGE_ANGLES)
                ])
                proj = verts @ n_dirs.T  # (6, 6)
                max_s = max(max_s, proj.max())
            s_required = 2 * max_s / np.sqrt(3)
            if s_required < best_s_local:
                best_s_local = s_required
                best_a_local = init_a
        return best_s_local, best_a_local

    # Phase 1: Optimize outer hexagon angle and scale
    s, outer_angle = _compute_s_and_angle(centers, angles)

    # Phase 2: Local optimization of inner positions
    def _objective(x):
        if time.time() > deadline:
            return np.inf
        cx = x[:24].reshape(12, 2)
        an = x[24:36]
        # Compute required s for current outer angle (fixed during inner optimization)
        a_rad = np.deg2rad(outer_angle)
        n_dirs = a_rad + np.pi/6 + np.arange(6) * np.pi/3
        max_s = 0
        for i in range(12):
            cx_i, cy_i = cx[i]
            a_i = np.deg2rad(an[i])
            verts = np.column_stack([
                cx_i + np.cos(a_i + _HEX_EDGE_ANGLES),
                cy_i + np.sin(a_i + _HEX_EDGE_ANGLES)
            ])
            proj = verts @ n_dirs.T
            max_s = max(max_s, proj.max())
        s_req = 2 * max_s / np.sqrt(3)
        # Check separation
        min_sep = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                # Distance between centers
                d = np.linalg.norm(cx[i] - cx[j])
                # Minimum distance for non-overlapping hexagons depends on relative angle
                # Approximate: need d >= 2.0 for axis-aligned
                # For SAT, use center distance as proxy
                if d < min_sep:
                    min_sep = d
        # Penalty for overlap: if any pair too close
        sep_penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(cx[i] - cx[j])
                if d < 2.0:
                    sep_penalty += (2.0 - d) ** 2
        return s_req + 100 * sep_penalty

    x0 = np.concatenate([centers.flatten(), angles])
    for restart in range(3):
        if time.time() > deadline:
            break
        # Add small perturbation for restarts
        if restart == 0:
            x_init = x0
        else:
            x_init = x0 + 0.1 * np.random.randn(len(x0))
        res = minimize(_objective, x_init, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000})
        if res.fun < np.inf:
            cx_opt = res.x[:24].reshape(12, 2)
            an_opt = res.x[24:36]
            s_opt, a_opt = _compute_s_and_angle(cx_opt, an_opt)
            # Validate
            inner_data = np.column_stack([cx_opt, an_opt])
            outer_data = np.array([0.0, 0.0, a_opt])
            valid, _, _ = _validate(inner_data, outer_data, s_opt)
            if valid and s_opt < best_s:
                best_s = s_opt
                best_inner = inner_data.copy()
                best_outer = outer_data.copy()
                best_valid = True

    # Phase 3: Joint optimization of outer angle
    if best_valid and time.time() < deadline:
        def _joint_objective(params):
            s_try = params[0]
            a_try = params[1]
            if time.time() > deadline:
                return np.inf
            a_rad = np.deg2rad(a_try)
            n_dirs = a_rad + np.pi/6 + np.arange(6) * np.pi/3
            max_s = 0
            for i in range(12):
                cx_i, cy_i = best_inner[i, 0], best_inner[i, 1]
                a_i = np.deg2rad(best_inner[i, 2])
                verts = np.column_stack([
                    cx_i + np.cos(a_i + _HEX_EDGE_ANGLES),
                    cy_i + np.sin(a_i + _HEX_EDGE_ANGLES)
                ])
                proj = verts @ n_dirs.T
                max_s = max(max_s, proj.max())
            s_req = 2 * max_s / np.sqrt(3)
            if s_try < s_req:
                return (s_req - s_try) ** 2 + s_try
            return s_try

        res = minimize(_joint_objective, [best_s, best_outer[2]], method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
        s_joint = max(res.x[0], np.sqrt(12))
        a_joint = res.x[1]
        inner_data = best_inner.copy()
        outer_data = np.array([0.0, 0.0, a_joint])
        valid, _, _ = _validate(inner_data, outer_data, s_joint)
        if valid and s_joint < best_s:
            best_s = s_joint
            best_outer = outer_data.copy()
            best_valid = True

    # Fallback
    if not best_valid:
        inner_data = np.column_stack([centers, angles])
        outer_data = np.array([0.0, 0.0, 0.0])
        s, outer_angle = _compute_s_and_angle(centers, angles)
        return inner_data, outer_data, s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
