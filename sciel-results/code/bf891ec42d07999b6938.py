"""Combined approach: exhaustive combinatorial search over hex-lattice topologies
(C(12,4)=495 second-ring arrangements) + SDF repulsive polishing + LP enclosing hexagon.

Structural idea (from inspiration): enumerate many second-ring topologies rather than
a fixed handful of hand-picked layouts, to discover compact non-lattice clusters.
Refinement machinery (from current program): vectorized LP for exact min-enclosing
hexagon, multi-pass angle sweeps, Nelder-Mead polish, high-precision grid.
New addition: SDF-based repulsive gradient descent to let hexagons relax into a
dense equilibrium before container optimization.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
from itertools import combinations

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin within 150s limit
_SQRT3 = math.sqrt(3)
_EPS = 1e-9


def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    angles = angle_rad + np.arange(6) * math.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _compute_all_vertices(inner_hexes):
    """Precompute all 6N vertices of inner hexagons as a numpy array (6N, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        verts[idx * 6: idx * 6 + 6] = _hex_vertices(ix, iy, iangle)
    return verts


def _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg):
    """For a fixed outer hexagon angle, find optimal center + side via LP.
    Returns (cx, cy, side) or None on failure.
    """
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return (0.0, 0.0, 0.0)

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

    v_dot_n = all_vertices @ normals.T  # (n_pts, 6)

    A_rows = []
    b_vals = []
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        # -n.c - R <= -v.n   and   n.c - R <= v.n
        A_rows.append(np.column_stack([np.full(n_pts, -nx), np.full(n_pts, -ny), np.full(n_pts, -1.0)]))
        b_vals.append(-v_dots)
        A_rows.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), np.full(n_pts, -1.0)]))
        b_vals.append(v_dots)

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return (cx, cy, side)
    except Exception:
        pass

    # Fallback: centroid + max projection
    cx_mean = all_vertices[:, 0].mean()
    cy_mean = all_vertices[:, 1].mean()
    centered = all_vertices - np.array([cx_mean, cy_mean])
    projs = centered @ normals.T
    R_opt = projs.max()
    if R_opt < 0:
        R_opt = 0.0
    side = R_opt / (_SQRT3 / 2)
    return (cx_mean, cy_mean, side)


def _enclosing_side_for_angle(all_vertices, outer_angle_deg):
    """Scalar side for a given angle (for objective functions)."""
    res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    if res:
        return res[2]
    return 1e6


def _min_enclosing_hex_side(inner_hexes):
    """Full multi-pass search for the minimum enclosing hexagon.
    Returns (center_x, center_y, side, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    all_vertices = _compute_all_vertices(inner_hexes)

    # Pass 1: coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res and res[2] < best_side:
            best_side = res[2]
            best_center = (res[0], res[1])
            best_angle = float(angle_deg)

    # Pass 2: fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res and res[2] < best_side:
                best_side = res[2]
                best_center = (res[0], res[1])
                best_angle = float(angle_deg)

    # Pass 3: even finer
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res and res[2] < best_side:
                best_side = res[2]
                best_center = (res[0], res[1])
                best_angle = float(angle_deg)

    # Pass 4: Nelder-Mead polish on (cx, cy, angle)
    if best_side < float('inf') and time.time() < _DEADLINE - 12.0:
        def objective(params):
            ang_p = float(params[2] % 60.0)
            return _enclosing_side_for_angle(all_vertices, ang_p)

        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.success:
                ang_nm = float(res.x[2] % 60.0)
                cx_v, cy_v, side_v = _min_enclosing_hex_side_lp(all_vertices, ang_nm)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_nm
        except Exception:
            pass

    # Pass 5: high-precision grid
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        window = 0.1
        steps = 150
        angles_hp = np.linspace(best_angle - window, best_angle + window, steps)
        for ang in angles_hp:
            ang_mod = float(ang % 60.0)
            res = _min_enclosing_hex_side_lp(all_vertices, ang_mod)
            if res and res[2] < best_side:
                best_side = res[2]
                best_center = (res[0], res[1])
                best_angle = ang_mod

    return best_center[0], best_center[1], best_side, best_angle


# ---------------------------------------------------------------------------
# Hex-lattice topology enumeration (structural idea from inspiration)
# ---------------------------------------------------------------------------

def _lattice_positions(n_ring=2, max_second=12):
    """Return lists of (i, j) axial coords for ring 0, ring 1, ring 2."""
    def axial_dist(i, j):
        return (abs(i) + abs(j) + abs(i + j)) // 2

    ring0 = [(0, 0)]
    ring1 = []
    ring2 = []
    for i in range(-n_ring, n_ring + 1):
        for j in range(-n_ring, n_ring + 1):
            d = axial_dist(i, j)
            if d == 1:
                ring1.append((i, j))
            elif d == 2:
                ring2.append((i, j))
    # Keep at most max_second ring-2 candidates (there are 12 for n_ring=2)
    ring2 = ring2[:max_second]
    return ring0, ring1, ring2


def _axial_to_xy(i, j, scale=1.0000001):
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    x = i * u[0] + j * v[0]
    y = i * u[1] + j * v[1]
    return (x, y)


def _build_topologies():
    """Enumerate topologies: center + 6 ring-1 + 4 chosen from 12 ring-2.
    Returns list of lists of (x, y) positions (11 each)."""
    ring0, ring1, ring2 = _lattice_positions()
    base = [ _axial_to_xy(i, j) for (i, j) in ring0 + ring1 ]  # 7 positions
    topologies = []
    for combo in combinations(range(len(ring2)), 4):
        extra = [ _axial_to_xy(ring2[k][0], ring2[k][1]) for k in combo ]
        topologies.append(base + extra)
    return topologies


# ---------------------------------------------------------------------------
# SDF-based repulsive relaxation (new differentiable idea)
# ---------------------------------------------------------------------------

def _hex_sdf_point(px, py, cx, cy, angle_deg):
    """Signed distance from point (px, py) to a unit regular hexagon.
    Positive outside, negative inside. Vectorized over points via caller.
    Returns (dist, grad_x, grad_y) for a single point.
    """
    angle_rad = math.radians(angle_deg)
    # Rotate point into hexagon local frame
    ca, sa = math.cos(-angle_rad), math.sin(-angle_rad)
    lx = ca * (px - cx) - sa * (py - cy)
    ly = sa * (px - cx) + ca * (py - cy)
    # Hexagon with vertices at radius 1, flat-top orientation:
    # sides at distance apothem = sqrt(3)/2 from center along 6 directions
    ap = _SQRT3 / 2
    # Compute signed distance to each of 6 edges, take max (for convex poly,
    # SDF = max over edges of (dot(p - v_edge_mid, n_edge) - ap) ... use half-planes)
    # Edge outward normals (flat-top hexagon):
    n_angles = np.array([0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
    nx = np.cos(n_angles)
    ny = np.sin(n_angles)
    # half-plane: n . (p) <= ap  ->  dist_k = ap - n . p  (positive inside)
    d = ap - (nx * lx + ny * ly)
    k = np.argmax(d)
    dist = -d[k]  # outside positive
    gxl = nx[k]
    gyl = ny[k]
    # Rotate gradient back to world frame
    gx = ca * gxl + sa * gyl
    gy = -sa * gxl + ca * gyl
    return dist, gx, gy


def _sdf_relax(positions, inner_angle=0.0, iterations=300, lr=0.02,
               min_gap=0.0):
    """Repulsive SDF gradient descent to relax positions into a denser packing.
    Each hexagon is pushed away from others when their SDFs indicate overlap
    (negative distance), and gently pulled toward the centroid to stay compact.
    Returns new positions.
    """
    pos = np.array(positions, dtype=float).copy()
    n = len(pos)
    for it in range(iterations):
        forces = np.zeros((n, 2))
        for a in range(n):
            for b in range(a + 1, n):
                # Sample points on hex a's boundary, check SDF into hex b
                # Use the 6 vertices of hex a as probes
                verts_a = _hex_vertices(pos[a, 0], pos[a, 1], inner_angle)
                for (px, py) in verts_a:
                    d, gx, gy = _hex_sdf_point(px, py, pos[b, 0], pos[b, 1], inner_angle)
                    if d < 0:  # overlap: push hex a away from b
                        # gradient w.r.t. point; to move hex a, use -grad direction
                        forces[a, 0] += -gx * (-d)
                        forces[a, 1] += -gy * (-d)
                        forces[b, 0] += gx * (-d)
                        forces[b, 1] += gy * (-d)
        # Gentle attraction to centroid to keep cluster compact
        centroid = pos.mean(axis=0)
        for a in range(n):
            forces[a, 0] += 0.02 * (centroid[0] - pos[a, 0])
            forces[a, 1] += 0.02 * (centroid[1] - pos[a, 1])
        pos += lr * forces
    return pos


def _try_sdf_polish(positions, inner_angle, time_budget=8.0):
    """Attempt SDF relaxation; return improved positions if it reduces enclosing side."""
    t0 = time.time()
    try:
        relaxed = _sdf_relax(positions, inner_angle, iterations=150, lr=0.01)
        # Only accept if it doesn't blow up
        if np.all(np.isfinite(relaxed)):
            return relaxed
    except Exception:
        pass
    return positions


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # ---- Phase 1: exhaustive combinatorial topology search ----
    # Enumerate C(12,4)=495 second-ring topologies. For each, do a quick
    # coarse enclosing-hexagon estimate to rank them, then fully optimize
    # the top-ranked ones.
    topologies = _build_topologies()

    # Quick ranking: coarse enclosing side for each topology (angle=0 and 30)
    ranked = []
    for idx, positions in enumerate(topologies):
        if time.time() > _DEADLINE - 20.0:
            break
        inner_candidate = [[pos[0], pos[1], 0.0] for pos in positions]
        all_vertices = _compute_all_vertices(inner_candidate)
        # Quick estimate at a few angles
        quick = min(_enclosing_side_for_angle(all_vertices, a)
                    for a in (0.0, 15.0, 30.0, 45.0))
        ranked.append((quick, idx))

    ranked.sort(key=lambda t: t[0])

    # Fully optimize the top K topologies
    K = 40
    for (score, idx) in ranked[:K]:
        if time.time() > _DEADLINE - 15.0:
            break
        positions = topologies[idx]
        for inner_angle in (0.0, 30.0):
            if time.time() > _DEADLINE - 15.0:
                break
            inner_candidate = [[pos[0], pos[1], inner_angle] for pos in positions]
            all_vertices = _compute_all_vertices(inner_candidate)

            # Coarse + fine angle sweep
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _min_enclosing_hex_side_lp(all_vertices, float(outer_angle_deg))
                if res and res[2] < best_coarse_side:
                    best_coarse_side = res[2]
                    best_coarse_angle = float(outer_angle_deg)
                    best_coarse_center = (res[0], res[1])

            if best_coarse_side < float('inf'):
                for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.0),
                                                 min(30.1, best_coarse_angle + 1.0), 0.05):
                    if time.time() > _DEADLINE - 15.0:
                        break
                    res = _min_enclosing_hex_side_lp(all_vertices, float(outer_angle_deg))
                    if res and res[2] < best_coarse_side:
                        best_coarse_side = res[2]
                        best_coarse_angle = float(outer_angle_deg)
                        best_coarse_center = (res[0], res[1])

            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = inner_candidate
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle

    # ---- Phase 2: SDF polish on the best configuration found ----
    if best_inner is not None and time.time() < _DEADLINE - 25.0:
        positions = [(p[0], p[1]) for p in best_inner]
        inner_angle = best_inner[0][2]
        relaxed = _try_sdf_polish(positions, inner_angle)
        relaxed_inner = [[p[0], p[1], inner_angle] for p in relaxed]
        all_vertices = _compute_all_vertices(relaxed_inner)
        # Quick check if relaxed version is better
        rel_side = min(_enclosing_side_for_angle(all_vertices, a)
                       for a in (best_outer_angle, (best_outer_angle + 15) % 60.0))
        if rel_side < best_side:
            best_inner = relaxed_inner
            best_side = rel_side

    # ---- Phase 3: full high-precision polish on the best configuration ----
    if best_inner is not None and time.time() < _DEADLINE - 10.0:
        cx, cy, side, angle = _min_enclosing_hex_side(best_inner)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle

    # Fallback
    if best_inner is None:
        positions = _build_topologies()[0]
        best_inner = [[pos[0], pos[1], 0.0] for pos in positions]
        cx, cy, best_side, best_outer_angle = _min_enclosing_hex_side(best_inner)
        best_center = (cx, cy)

    return best_inner, best_center, best_side, best_outer_angle
