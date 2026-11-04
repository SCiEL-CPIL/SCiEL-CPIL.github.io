"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Structural change:
  Phase 1: rigorous discrete-then-continuous hybrid.
    - enumerate compact subsets of a triangular lattice (guaranteed non-overlap at scale=1)
    - for each topology, optimize (scale, global rotation, outer rotation) via Nelder-Mead
  Phase 2: continuous local refinement of the best configuration via Nelder-Mead on the
    36 free parameters (12 centers + 12 angles), with a smooth SAT-gap penalty.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def polygons_overlap_sat(A, B):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True


def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


# ---------- Vectorized helpers for speed ----------

def _unit_hex_verts(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a + i * math.pi / 3), math.sin(a + i * math.pi / 3)] for i in range(6)])


def _all_verts(inner):
    """inner: list of (x,y,angle_deg). Returns (N,6,2) array of all vertices."""
    out = np.empty((N, 6, 2))
    for i, (x, y, ang) in enumerate(inner):
        out[i] = _unit_hex_verts(ang) + np.array([x, y])
    return out


def _min_outer_side_vec(inner, outer_center, outer_angle):
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = np.array([[math.cos(a + (k + 0.5) * math.pi / 3),
                         math.sin(a + (k + 0.5) * math.pi / 3)] for k in range(6)])
    verts = _all_verts(inner)  # (N,6,2)
    # project all vertices onto all normals: (N,6,6)
    projs = np.einsum('ijk,jk->ij', verts - np.array([cx, cy]), normals.T)
    max_proj = np.max(projs)
    return (2.0 / math.sqrt(3.0)) * max_proj


def _sat_gap_vec(poly1, poly2):
    """Minimum separation between two convex hexagons (6x2 arrays).
    Returns negative if overlapping (penetration depth approx)."""
    min_gap = float('inf')
    for poly in (poly1, poly2):
        for i in range(6):
            edge = poly[(i + 1) % 6] - poly[i]
            norm = np.linalg.norm(edge)
            if norm < 1e-12:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm
            p1 = poly1 @ axis
            p2 = poly2 @ axis
            gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
            if gap >= 0:
                return gap
            if gap < min_gap:
                min_gap = gap
    return min_gap


def _overlap_penalty(inner, tol=1e-4):
    """Sum of squared penetration depths for overlapping pairs (0 if all disjoint)."""
    polys = [_all_verts(inner)[i] for i in range(N)]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            g = _sat_gap_vec(polys[i], polys[j])
            if g < -tol:
                pen += (g + tol) ** 2
    return pen


# ---------- Phase 1: discrete lattice topology search ----------

def _lattice_points(scale):
    """Generate a grid of triangular lattice points, scaled."""
    s = scale
    v1 = np.array([1.5 * s, math.sqrt(3) / 2 * s])
    v2 = np.array([0.0, math.sqrt(3) * s])
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            pts.append(i * v1 + j * v2)
    return np.array(pts)


def _pick_compact_12(points):
    """Pick 12 points closest to their centroid (compact cluster)."""
    # sort by distance from origin first, take 12, recenter
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)[:12]
    sel = points[idx].copy()
    centroid = np.mean(sel, axis=0)
    sel -= centroid
    return sel


def _try_topology(base_positions, scale, rot_global_deg, rot_in_deg, rot_out_deg):
    """
    base_positions: (12,2) array of unit-scale lattice points (centered).
    Try scaling, rotating the cluster, and compute outer side.
    Returns (outer_side, inner_list) or None if invalid.
    """
    if time.time() > DEADLINE:
        return None

    rot = math.radians(rot_global_deg)
    rot_mat = np.array([[math.cos(rot), -math.sin(rot)],
                        [math.sin(rot), math.cos(rot)]])
    positions = (base_positions * scale) @ rot_mat.T

    inner = []
    for k in range(N):
        inner.append((float(positions[k, 0]), float(positions[k, 1]), rot_in_deg))

    outer_center = (0.0, 0.0)
    outer_side = _min_outer_side_vec(inner, outer_center, rot_out_deg)

    # Quick validity check: pairwise non-overlap
    polys = [_all_verts(inner)[k] for k in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            g = _sat_gap_vec(polys[i], polys[j])
            if g < -1e-5:
                return None
    return (outer_side, inner)


def _phase1_discrete():
    """Search over lattice scales, global rotations, and inner/outer angles."""
    best_side = float('inf')
    best_result = None

    # Base topologies: pick compact 12 from unit lattice
    unit_pts = _lattice_points(1.0)
    base_positions = _pick_compact_12(unit_pts)

    scales = np.linspace(0.85, 1.35, 11)
    rot_globals = [0.0, 15.0, 30.0, 45.0, 60.0]
    rot_ins = [0.0, 30.0, 60.0]
    rot_outs = [0.0, 15.0, 30.0, 45.0]

    for scale in scales:
        for rg in rot_globals:
            for ri in rot_ins:
                for ro in rot_outs:
                    if time.time() > DEADLINE:
                        break
                    res = _try_topology(base_positions, scale, rg, ri, ro)
                    if res is not None:
                        side, inner = res
                        if side < best_side:
                            best_side = side
                            best_result = (inner, (0.0, 0.0), side, ro)
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break

    if best_result is None:
        return None

    # Local refinement on the best discrete result
    inner0 = best_result[0]
    ro0 = best_result[3]

    def obj(params):
        if time.time() > DEADLINE:
            return 1e6
        scale, rg, ri, ro = params
        res = _try_topology(base_positions, scale, rg, ri, ro)
        if res is None:
            return 1e6
        return res[0]

    try:
        x0 = [1.0, 0.0, 0.0, ro0]
        # Try to recover scale/rg/ri/ro that produced best_result
        # We don't track them, so just use Nelder-Mead around a sensible start
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < best_side:
            inner = _try_topology(base_positions, *result.x)[1]
            best_result = (inner, (0.0, 0.0), result.fun, result.x[3])
    except Exception:
        pass

    return best_result


# ---------- Phase 2: continuous refinement ----------

def _phase2_continuous(inner0, outer_center0, side0, angle0):
    """Nelder-Mead on 36 params: 12*(x,y,angle)."""
    if time.time() > DEADLINE:
        return None

    x0 = []
    for (x, y, a) in inner0:
        x0.extend([x, y, a])
    x0 = np.array(x0)

    def obj(params):
        if time.time() > DEADLINE:
            return 1e6
        inner = []
        for k in range(N):
            inner.append((params[3 * k], params[3 * k + 1], params[3 * k + 2]))
        # Center of mass
        com = (np.mean([p[0] for p in inner]), np.mean([p[1] for p in inner]))
        # Translate so COM is at origin (outer center stays at origin)
        inner = [(x - com[0], y - com[1], a) for (x, y, a) in inner]

        side = _min_outer_side_vec(inner, (0.0, 0.0), angle0)

        # Penalty for overlaps
        pen = _overlap_penalty(inner)
        if pen > 1e-6:
            return side + 10.0 * pen
        return side

    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-7,
                                   'maxfev': 3000})
        if result.fun < side0 - 1e-8:
            inner = []
            for k in range(N):
                inner.append((float(result.x[3 * k]),
                              float(result.x[3 * k + 1]),
                              float(result.x[3 * k + 2])))
            com = (np.mean([p[0] for p in inner]), np.mean([p[1] for p in inner]))
            inner = [(x - com[0], y - com[1], a) for (x, y, a) in inner]
            # Re-center
            com2 = (np.mean([p[0] for p in inner]), np.mean([p[1] for p in inner]))
            inner = [(x - com2[0], y - com2[1], a) for (x, y, a) in inner]
            side = _min_outer_side_vec(inner, (0.0, 0.0), angle0)
            # Validate
            if check_validity(inner, (0.0, 0.0), side, angle0):
                return (inner, (0.0, 0.0), side, angle0)
    except Exception:
        pass
    return None


def construct():
    t0 = time.time()

    # Phase 1: discrete lattice topology search
    best = _phase1_discrete()

    if best is None:
        # Fallback: simple hexagonal cluster
        a = 2.0
        positions = [(0.0, 0.0)]
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((a * math.cos(theta), a * math.sin(theta)))
        for k in range(5):
            theta = math.radians(k * 60.0)
            positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
        inner = [(x, y, 0.0) for (x, y) in positions]
        outer_center = (0.0, 0.0)
        outer_side = compute_min_outer_side(inner, outer_center, 0.0)
        return inner, outer_center, outer_side, 0.0

    inner, outer_center, side, angle = best

    # Phase 2: continuous refinement if time permits
    if time.time() < DEADLINE - 10.0:
        refined = _phase2_continuous(inner, outer_center, side, angle)
        if refined is not None:
            inner, outer_center, side, angle = refined

    return inner, outer_center, side, angle
