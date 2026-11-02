"""Packing 12 unit regular hexagons in a minimal regular hexagon."""
import math
import time

N = 12

# Honeycomb cluster of 12 unit hexagons (rows -1,0,1 with 5,4,3 centers).
# Unit hexagon: circumradius = 1, inradius = sqrt(3)/2, horizontal pitch = 1.5, vertical pitch = sqrt(3).
# Rows offset horizontally by 0.75 to nest.
_SQ3 = math.sqrt(3.0)
_ROW_YS = [-_SQ3, 0.0, _SQ3]
_ROW_XS = {
    0: [0.75 * k for k in range(-2, 3)],          # 5 centers
    1: [0.75 + 0.75 * k for k in range(-2, 2)],   # 4 centers
    2: [0.75 * k for k in range(-1, 2)],          # 3 centers
}
_CENTER0 = [_ROW_XS[i][j] for i in range(3) for j in range(len(_ROW_XS[i]))]
_CENTER1 = [_ROW_YS[i] for i in range(3) for j in range(len(_ROW_XS[i]))]
# Cluster centroid (symmetric -> (0,0)).
_CX = sum(_CENTER0) / N
_CY = sum(_CENTER1) / N
# Centered local centers
_LOCAL = [(_CENTER0[i] - _CX, _CENTER1[i] - _CY) for i in range(N)]


def _rot_pt(x, y, c, s):
    return (x * c - y * s, x * s + y * c)


def _hexagon_vertices(cx, cy, R, ang):
    """Vertices of a regular hexagon, circumradius R, center (cx,cy), rotation ang (rad)."""
    vx, vy = [], []
    for k in range(6):
        a = ang + k * (math.pi / 3.0)
        vx.append(cx + R * math.cos(a))
        vy.append(cy + R * math.sin(a))
    return vx, vy


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


def _point_in_hex(px, py, cx, cy, R, ang):
    """Robust point-in-convex-hexagon test via winding/edge consistency."""
    vx, vy = _hexagon_vertices(cx, cy, R, ang)
    prev_x, prev_y = vx[5], vy[5]
    for k in range(6):
        x1, y1 = vx[k], vy[k]
        x2, y2 = vx[(k + 1) % 6], vy[(k + 1) % 6]
        # vector from vertex to point and edge
        if _cross2(x1 - px, y1 - py, x2 - x1, y2 - y1) < -1e-9:
            return False
    return True


def _hex_hex_overlap(c1, a1, c2, a2, eps=1e-7):
    """Check if two unit hexagons (centers, rotations) overlap (interior)."""
    dx, dy = c1[0] - c2[0], c1[1] - c2[1]
    d2 = dx * dx + dy * dy
    # max possible distance between centers before touching is 2*circumradius=2
    if d2 > (2.0 - eps) ** 2:
        return False
    # Separation axis: 6 edge normals of hex1 + 6 of hex2
    for (cx, cy, ang) in ((c1, a1), (c2, a2)):
        for k in range(6):
            a = ang + k * (math.pi / 3.0)
            # edge normal (inward) direction
            nx, ny = -math.sin(a), math.cos(a)
            # project both hexes onto axis
            p1, p2 = [], []
            for j in range(6):
                aa = ang + j * (math.pi / 3.0)
                p1.append((cx + math.cos(aa)) * nx + (cy + math.sin(aa)) * ny)
                aa2 = (a2 if (cx, cy) == (c1, a1) else a1)
                p2.append((c2[0] + math.cos(aa2 + (j - k) * 0)) )  # placeholder
            # recompute properly
            p1 = [(cx + math.cos(ang + j * (math.pi / 3.0))) * nx + (cy + math.sin(ang + j * (math.pi / 3.0))) * ny for j in range(6)]
            p2 = [(c2[0] + math.cos(a2 + j * (math.pi / 3.0))) * nx + (c2[1] + math.sin(a2 + j * (math.pi / 3.0))) * ny for j in range(6)]
            lo1, hi1 = min(p1), max(p1)
            lo2, hi2 = min(p2), max(p2)
            if hi1 + eps < lo2 or hi2 + eps < lo1:
                return False  # found separating axis -> no overlap
    return True


def _valid_cluster(rot, eps=1e-7):
    """Check no two cluster hexes overlap under global rotation rot."""
    c, s = math.cos(rot), math.sin(rot)
    pos = []
    for (lx, ly) in _LOCAL:
        pos.append(_rot_pt(lx, ly, c, s))
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_hex_overlap(pos[i], rot, pos[j], rot, eps):
                return False
    return True


def _min_side_for(rot, outer_ang, eps=1e-9):
    """Minimum outer side length S such that all cluster hexes (rotation rot)
    fit inside outer hexagon (center 0,0, rotation outer_ang) of side S.
    Outer circumradius = S."""
    c, s = math.cos(rot), math.sin(rot)
    # all vertices of all inner hexes
    verts = []
    for (lx, ly) in _LOCAL:
        rx, ry = _rot_pt(lx, ly, c, s)
        for k in range(6):
            a = rot + k * (math.pi / 3.0)
            vx = rx + math.cos(a)
            vy = ry + math.sin(a)
            verts.append((vx, vy))
    # outer hexagon edges: 6 constraints. Edge k has outward normal at angle outer_ang + k*pi/3
    # A point is inside if for all edges, dot(p, n) <= R where R = circumradius = S
    # (since for a regular hexagon the apothem = S*sqrt(3)/2 and the support in normal dir = S)
    # Actually support function of regular hexagon (circumradius S) in direction n = S*max over vertices dot(v,S_hat)
    # = S * (cos of angle between n and nearest vertex). The max |dot| = S when n aligns with a vertex.
    # Constraint: for each edge normal n_k, max_v dot(v, n_k) <= S * h(n_k) where h is support of unit hex.
    # Simpler: use the 6 edge-half-plane form. Edge outward normal direction theta_k = outer_ang + k*pi/3.
    # Half-plane: dot(p, (cos theta_k, sin theta_k)) <= S * cos(pi/6) = S*sqrt(3)/2? 
    # For a regular hexagon with circumradius S, apothem = S*sqrt(3)/2, and the edge normal points outward
    # with the edge at distance apothem from center. So constraint: dot(p, n_k) <= S*sqrt(3)/2.
    S = 0.0
    for k in range(6):
        theta = outer_ang + k * (math.pi / 3.0)
        nx, ny = math.cos(theta), math.sin(theta)
        m = 0.0
        for (vx, vy) in verts:
            d = vx * nx + vy * ny
            if d > m:
                m = d
        # need m <= S*sqrt(3)/2  => S >= m / (sqrt(3)/2)
        req = m / (_SQ3 / 2.0)
        if req > S:
            S = req
    return S


def _objective(theta, phi):
    """theta: cluster rotation, phi: outer rotation. Returns outer side length (inf if invalid)."""
    if not _valid_cluster(theta):
        return float('inf')
    return _min_side_for(theta, phi)


def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin under 120s

    best_S = float('inf')
    best_theta = 0.0
    best_phi = 0.0

    # Phase 1: coarse grid over cluster rotation theta in [0, pi/3) (60 deg symmetry of hex).
    # For each theta, find best phi in [0, pi/3).
    def search_grid(n_theta, n_phi):
        nonlocal best_S, best_theta, best_phi
        for i in range(n_theta):
            theta = (i + 0.5) * (math.pi / 3.0) / n_theta
            if not _valid_cluster(theta):
                continue
            # find best phi
            for j in range(n_phi):
                phi = (j + 0.5) * (math.pi / 3.0) / n_phi
                S = _min_side_for(theta, phi)
                if S < best_S:
                    best_S = S
                    best_theta = theta
                    best_phi = phi
            if time.time() - t0 > DEADLINE - 30:
                return

    search_grid(30, 30)

    # Phase 2: local refinement (golden-section) around best theta and phi.
    def refine(center_theta, center_phi, span_theta, span_phi, iters=20):
        nonlocal best_S, best_theta, best_phi
        gr = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        # refine theta (fix phi)
        lo, hi = center_theta - span_theta, center_theta + span_theta
        for _ in range(iters):
            if time.time() - t0 > DEADLINE - 10:
                break
            m1 = lo + gr * (hi - lo)
            m2 = hi - gr * (hi - lo)
            f1 = _objective(m1, center_phi)
            f2 = _objective(m2, center_phi)
            if f1 < f2:
                hi = m2
            else:
                lo = m1
        theta_star = (lo + hi) / 2.0
        # refine phi (fix theta_star)
        lo, hi = center_phi - span_phi, center_phi + span_phi
        for _ in range(iters):
            if time.time() - t0 > DEADLINE - 5:
                break
            m1 = lo + gr * (hi - lo)
            m2 = hi - gr * (hi - lo)
            f1 = _objective(theta_star, m1)
            f2 = _objective(theta_star, m2)
            if f1 < f2:
                hi = m2
            else:
                lo = m1
        phi_star = (lo + hi) / 2.0
        S = _objective(theta_star, phi_star)
        if S < best_S:
            best_S = S
            best_theta = theta_star
            best_phi = phi_star

    # Refine in a few neighborhoods (to avoid local minima due to periodicity)
    for k in range(6):
        ct = (k * (math.pi / 3.0) + best_theta) % (math.pi / 3.0)
        cp = (k * (math.pi / 3.0) + best_phi) % (math.pi / 3.0)
        refine(ct, cp, (math.pi / 3.0) / 30.0, (math.pi / 3.0) / 30.0, iters=18)
        if time.time() - t0 > DEADLINE - 5:
            break

    # Build inner positions at best_theta
    c, s = math.cos(best_theta), math.sin(best_theta)
    inner = []
    for (lx, ly) in _LOCAL:
        rx, ry = _rot_pt(lx, ly, c, s)
        inner.append([rx, ry, math.degrees(best_theta)])

    if math.isinf(best_S):
        # fallback: trivial large hexagon
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), best_S, math.degrees(best_phi)
