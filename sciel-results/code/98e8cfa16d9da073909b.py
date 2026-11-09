"""Packing 12 unit hexagons in a minimal regular hexagon via gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

# Precompute unit hexagon vertices (flat-top, circumradius 1)
_PHI = np.pi / 3.0
_UV = np.column_stack([np.cos(_PHI * np.arange(6)), np.sin(_PHI * np.arange(6))])
_UV6 = np.tile(_UV, (6, 1))
_EDGES = np.array([
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1)
])
_EDGES6 = np.tile(_EDGES, (6, 1))
_E0 = _EDGES6[:, 0]
_E1 = _EDGES6[:, 1]


def _hex_verts(cx, cy, ang):
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return cx + _UV @ R.T, cy + _UV @ R.T


def _rot_verts(cx, cy, ang):
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return cx + _UV6 @ R.T, cy + _UV6 @ R.T


def _seg_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-14:
        return False
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / denom
    return -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9


def _cross2(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _point_in_hex(px, py, vx, vy):
    for i in range(6):
        if _cross2(vx[i], vy[i], vx[(i + 1) % 6], vy[(i + 1) % 6], px, py) < 0:
            return False
    return True


def _hex_overlap(ax, ay, bx, by, cxi, cyi, cj, cjy, ang_i, ang_j):
    """Check if hex i (centered ax,ay, angle ang_i) overlaps hex j (centered bx,by, angle ang_j)."""
    # Quick rejection: center distance > 2 * circumradius
    dx, dy = bx - ax, by - ay
    if dx * dx + dy * dy > 4.5:
        return False
    # Check vertex containment
    vi_x, vi_y = _hex_verts(ax, ay, ang_i)
    vj_x, vj_y = _hex_verts(bx, by, ang_j)
    for k in range(6):
        if _point_in_hex(vi_x[k], vi_y[k], vj_x, vj_y):
            return True
        if _point_in_hex(vj_x[k], vj_y[k], vi_x, vi_y):
            return True
    # Check edge intersections
    for a, b in _EDGES:
        for c, d in _EDGES:
            if _seg_intersect(vi_x[a], vi_y[a], vi_x[b], vi_y[b],
                              vj_x[c], vj_y[c], vj_x[d], vj_y[d]):
                return True
    return False


def _hex_in_outer(cx, cy, ang, ox, oy, L, oang):
    """Check if unit hexagon at (cx,cy) angle ang is inside outer hexagon (ox,oy,L,oang)."""
    vi_x, vi_y = _hex_verts(cx, cy, ang)
    # Outer hexagon vertices
    ov_x = ox + L * np.cos(oang + _PHI * np.arange(6))
    ov_y = oy + L * np.sin(oang + _PHI * np.arange(6))
    for k in range(6):
        if not _point_in_hex(vi_x[k], vi_y[k], ov_x, ov_y):
            return False
    return True


def _objective(params, beta):
    """Objective: outer side length + penalty for violations."""
    n = N
    # Parse params: [x0,y0,a0, x1,y1,a1, ..., xn-1,yn-1,an-1, ox,oy,L,oang]
    pos = params[:3 * n].reshape(n, 3)
    ox, oy, L, oang = params[3 * n:3 * n + 4]

    obj = L

    # Overlap penalty (soft)
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_overlap(pos[i, 0], pos[i, 1], pos[j, 0], pos[j, 1],
                            0, 0, 0, 0, pos[i, 2], pos[j, 2]):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                d = np.sqrt(dx * dx + dy * dy)
                # Penalty proportional to overlap depth
                overlap_depth = max(0, 2.0 - d)
                obj += beta * overlap_depth * overlap_depth

    # Containment penalty
    for i in range(n):
        if not _hex_in_outer(pos[i, 0], pos[i, 1], pos[i, 2], ox, oy, L, oang):
            # Compute max distance of hex vertices from outer boundary
            vi_x, vi_y = _hex_verts(pos[i, 0], pos[i, 1], pos[i, 2])
            ov_x = ox + L * np.cos(oang + _PHI * np.arange(6))
            ov_y = oy + L * np.sin(oang + _PHI * np.arange(6))
            # Approximate penalty: how far outside
            max_violation = 0.0
            for k in range(6):
                # Distance from vertex to nearest outer edge
                min_dist = 1e10
                for e in range(6):
                    ex1, ey1 = ov_x[e], ov_y[e]
                    ex2, ey2 = ov_x[(e + 1) % 6], ov_y[(e + 1) % 6]
                    # Signed distance
                    ex, ey = ex2 - ex1, ey2 - ey1
                    elen = np.sqrt(ex * ex + ey * ey)
                    if elen < 1e-14:
                        continue
                    # Project point onto edge
                    t = ((vi_x[k] - ex1) * ex + (vi_y[k] - ey1) * ey) / (elen * elen)
                    t = max(0, min(1, t))
                    px, py = ex1 + t * ex, ey1 + t * ey
                    dist = np.sqrt((vi_x[k] - px) ** 2 + (vi_y[k] - py) ** 2)
                    # Check if inside
                    if _point_in_hex(vi_x[k], vi_y[k], ov_x, ov_y):
                        min_dist = min(min_dist, dist)
                    else:
                        min_dist = min(min_dist, dist)
                        max_violation = max(max_violation, dist)
            if max_violation > 0:
                obj += beta * max_violation * max_violation

    return obj


def _initial_grid():
    """Create initial 2x6 grid of hexagons."""
    inner = []
    for row in range(2):
        for col in range(6):
            x = col * 2.0 - 5.0  # center the grid
            y = row * np.sqrt(3) - np.sqrt(3) / 2
            inner.append([x, y, 0.0])
    return inner


def construct():
    global _START, _DEADLINE

    best_score = None
    best_result = None

    # Initial grid configuration
    grid = _initial_grid()
    # Compute initial outer hexagon
    # Grid spans x: -6 to 6, y: -sqrt(3)/2 to sqrt(3)/2
    # Width = 12, Height = sqrt(3)
    # For flat-top outer hex with flat side facing the long axis:
    # The support in x-direction needs L such that the hex covers x in [-6,6]
    # For a regular hexagon with side L, flat-top, the x-extent is L (from vertex to vertex)
    # Actually for flat-top: vertices at angles 0,60,120,180,240,300
    # x-extent = L (from vertex at 0 to vertex at 180)
    # y-extent = L * sqrt(3) (from flat top to flat bottom)
    # We need x-extent >= 12, y-extent >= sqrt(3)
    # So L >= 12 from x, L >= 1 from y
    # L = 12, oang = 0 (flat-top)
    # But wait, let me reconsider. The grid is wider than tall.
    # If we orient the outer hex with a flat side perpendicular to the x-axis:
    # That means vertices at 30, 90, 150, 210, 270, 330 degrees
    # x-extent = L * cos(30) * 2 = L * sqrt(3)
    # y-extent = L * 2
    # Need L * sqrt(3) >= 12 => L >= 12/sqrt(3) = 4*sqrt(3) ≈ 6.928
    # Need L * 2 >= sqrt(3) => L >= sqrt(3)/2 ≈ 0.866
    # So L = 4*sqrt(3) ≈ 6.928, oang = 30 degrees
    # Hmm, let me just try both orientations and see.

    # Try orientation with flat side facing x-axis (oang = 30 deg = pi/6)
    # This gives x-extent = L*sqrt(3), y-extent = 2L
    # For L = 4*sqrt(3): x-extent = 12, y-extent = 8*sqrt(3) ≈ 13.86
    # That's way more than needed in y.

    # Try oang = 0 (flat-top): x-extent = 2L, y-extent = L*sqrt(3)
    # For L = 6: x-extent = 12, y-extent = 6*sqrt(3) ≈ 10.39
    # Still more than needed in y.

    # The minimal L is determined by the constraint that all hex vertices
    # must be inside. Let me just set L = 6.0 initially and let optimization work.

    ox, oy = 0.0, 0.0
    L_init = 6.0
    oang_init = 0.0

    # Build initial parameter vector
    params = []
    for x, y, a in grid:
        params.extend([x, y, a])
    params.extend([ox, oy, L_init, oang_init])
    params = np.array(params)

    # Try a few different initial configurations and betas
    configs = [
        (0.0, 0.0, 6.0, 0.0, 50.0),   # flat-top, L=6, beta=50
        (0.0, 0.0, 6.0, np.pi/6, 50.0),  # pointy-top, L=6, beta=50
        (0.0, 0.0, 7.0, 0.0, 50.0),
        (0.0, 0.0, 7.0, np.pi/6, 50.0),
    ]

    for ox0, oy0, L0, oang0, beta in configs:
        if time.time() > _DEADLINE:
            break

        p0 = []
        for x, y, a in grid:
            p0.extend([x, y, a])
        p0.extend([ox0, oy0, L0, oang0])
        p0 = np.array(p0)

        try:
            res = minimize(
                _objective, p0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
        except Exception:
            continue

        if res.success or res.fun < (best_score if best_score else 1e10):
            p = res.x
            inner = []
            for i in range(N):
                inner.append([float(p[3*i]), float(p[3*i+1]), float(np.degrees(p[3*i+2]))])
            ox_r, oy_r, L_r, oang_r = float(p[3*N]), float(p[3*N+1]), float(p[3*N+2]), float(p[3*N+3])
            # Validate: check no overlaps and containment
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if _hex_overlap(inner[i][0], inner[i][1], inner[j][0], inner[j][1],
                                    0, 0, 0, 0, np.radians(inner[i][2]), np.radians(inner[j][2])):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                for i in range(N):
                    if not _hex_in_outer(inner[i][0], inner[i][1], np.radians(inner[i][2]),
                                         ox_r, oy_r, L_r, oang_r):
                        valid = False
                        break
            if valid and L_r < (best_score if best_score else 1e10):
                best_score = L_r
                best_result = (inner, (ox_r, oy_r), L_r, np.degrees(oang_r))

    # If optimization didn't find a valid solution, return the grid with a safe outer
    if best_result is None:
        inner = [[float(x), float(y), float(a)] for x, y, a in grid]
        # Compute a safe outer hexagon
        # Find bounding box of all hex vertices
        max_x, min_x = -1e10, 1e10
        max_y, min_y = -1e10, 1e10
        for x, y, a in grid:
            vx, vy = _hex_verts(x, y, a)
            max_x = max(max_x, np.max(vx))
            min_x = min(min_x, np.min(vx))
            max_y = max(max_y, np.max(vy))
            min_y = min(min_y, np.min(vy))
        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        # For flat-top hex: x-extent = 2L, y-extent = L*sqrt(3)
        # L >= width/2 and L >= height/sqrt(3)
        L = max(width / 2, height / np.sqrt(3)) * 1.1  # 10% margin
        return inner, (cx, cy), float(L), 0.0

    return best_result
