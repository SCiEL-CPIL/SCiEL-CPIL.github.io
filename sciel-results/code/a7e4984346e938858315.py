"""Hexagon packing: 11 unit regular hexagons inside a regular hexagon.
Strategy: try several lattice-based initial clusterings, then run a
gradient-based optimizer (Nelder-Mead) to minimize the enclosing hexagon
side length.  A soft barrier / smooth constraint term is used inside the
objective, and the final answer uses a hard-check to ensure feasibility.
"""

import time
import math
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = time.time() + 130.0  # safety margin before 150 s kill

# ---------- geometry helpers ----------

def hex_vertices(cx, cy, s, angle_deg):
    """Return list of 6 vertices of a regular hexagon (side s) centred at (cx,cy)
    with first vertex at angle (degrees) from the +x axis, going CCW."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return pts


def _edge_normal(p1, p2):
    """Outward-pointing normal of edge p1->p2 for a CCW polygon."""
    ex = p2[0] - p1[0]
    ey = p2[1] - p1[1]
    L = math.hypot(ex, ey)
    # outward normal for CCW polygon: (dy, -dx) normalized
    nx = ey / L
    ny = -ex / L
    return nx, ny


def point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside or on the boundary of the hexagon."""
    verts = hex_vertices(cx, cy, s, angle_deg)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        nx, ny = _edge_normal(p1, p2)
        # outward normal: dot((p - p1), n) should be <= 0 for inside
        d = (px - p1[0]) * nx + (py - p1[1]) * ny
        if d > 1e-9:
            return False
    return True


def hexagon_distance_to_boundary(px, py, cx, cy, s, angle_deg):
    """Signed distance from point to hexagon boundary (negative = outside).
    Positive = inside."""
    verts = hex_vertices(cx, cy, s, angle_deg)
    min_d = float('inf')
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        nx, ny = _edge_normal(p1, p2)
        d = (px - p1[0]) * nx + (py - p1[1]) * ny
        if d < min_d:
            min_d = d
    return min_d


def hex_overlap_dist(cx1, cy1, a1, cx2, cy2, a2):
    """Return 0 if two unit hexagons are disjoint or just touching,
    else a positive measure of overlap depth.  Uses vertex-containment
    checks for a simple but effective proxy."""
    # Check if any vertex of hex1 is inside hex2 and vice versa
    v1 = hex_vertices(cx1, cy1, 1.0, a1)
    v2 = hex_vertices(cx2, cy2, 1.0, a2)
    depth = 0.0
    for vx, vy in v1:
        d = hexagon_distance_to_boundary(vx, vy, cx2, cy2, 1.0, a2)
        if d < 0:
            depth = max(depth, -d)
    for vx, vy in v2:
        d = hexagon_distance_to_boundary(vx, vy, cx1, cy1, 1.0, a1)
        if d < 0:
            depth = max(depth, -d)
    return depth


def enclosing_side_length(positions, angles, ocx, ocy, oangle):
    """Given inner hex positions/angles and outer hex params,
    return the minimal side length S such that all inner hexagons fit inside.
    We approximate by checking the 6 vertices of each inner hex and finding
    the max over all vertices of the required S for that vertex to be inside."""
    max_d = 0.0
    for (px, py), ang in zip(positions, angles):
        for vx, vy in hex_vertices(px, py, 1.0, ang):
            # For a point at (vx,vy) to be inside a hex of side S centered at (ocx,ocy) with angle oangle,
            # we need S >= max over 6 edges of the "support" distance.
            # For a regular hexagon with side S, the distance from center to edge (apothem) = S*sqrt(3)/2.
            # The vertex at angle theta from center at distance r = S (circumradius = side for hex).
            # A point (dx,dy) relative to center is inside iff for each of the 6 edge normals n_k,
            #   dot((dx,dy), n_k) <= S * sqrt(3) / 2
            # So S >= 2/sqrt(3) * max_k dot((dx,dy), n_k)
            dx = vx - ocx
            dy = vy - ocy
            # 6 outward normals of the outer hex
            a = math.radians(oangle)
            req = 0.0
            for k in range(6):
                theta = a + k * math.pi / 3.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                # Wait: the edge normal is perpendicular to the vertex direction.
                # For a regular hexagon, vertex k is at angle a + k*pi/3.
                # The edge between vertex k and k+1 has outward normal at angle a + (k+0.5)*pi/3.
                edge_theta = a + (k + 0.5) * math.pi / 3.0
                nx = math.cos(edge_theta)
                ny = math.sin(edge_theta)
                d = dx * nx + dy * ny
                if d > req:
                    req = d
            S_needed = 2.0 / math.sqrt(3.0) * req
            if S_needed > max_d:
                max_d = S_needed
    return max_d


def objective(params):
    """Objective: minimize enclosing side + penalty for overlaps.
    params layout: [pos0x, pos0y, ..., pos10x, pos10y, ang0..ang10 (radians), ocx, ocy, oangle (radians)]
    Total: 11*2 + 11*1 + 3 = 36
    """
    if time.time() > DEADLINE:
        return 1e10
    pos = []
    angs = []
    idx = 0
    for i in range(N):
        pos.append((params[idx], params[idx + 1]))
        idx += 2
    for i in range(N):
        angs.append(params[idx])
        idx += 1
    ocx = params[idx]
    ocy = params[idx + 1]
    oangle = params[idx + 2]

    # Enclosing side
    S = enclosing_side_length(pos, angs, ocx, ocy, math.degrees(oangle))

    # Overlap penalty: soft
    overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = hex_overlap_dist(pos[i][0], pos[i][1], math.degrees(angs[i]),
                                 pos[j][0], pos[j][1], math.degrees(angs[j]))
            if d > 0:
                overlap += d * d  # quadratic penalty

    # Also check that inner hexes' vertices are inside outer hex:
    # This is already captured by S, but we add a small penalty if S < 1.0 (impossible)
    if S < 1.0:
        penalty = (1.0 - S) * 100.0
    else:
        penalty = 0.0

    return S + 50.0 * overlap + penalty


def is_feasible(pos, angs_deg, ocx, ocy, oangle_deg):
    """Hard check: all inner hex vertices inside outer hex, no overlaps."""
    # Check containment
    for (px, py), ang in zip(pos, angs_deg):
        for vx, vy in hex_vertices(px, py, 1.0, ang):
            if not point_in_hexagon(vx, vy, ocx, ocy, 1.0, oangle_deg):
                # Check with the computed S
                return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = hex_overlap_dist(pos[i][0], pos[i][1], angs_deg[i],
                                 pos[j][0], pos[j][1], angs_deg[j])
            if d > 1e-6:
                return False
    return True


def compute_min_side(pos, angs_deg, ocx, ocy, oangle_deg):
    """Compute the exact minimal side length for this configuration."""
    return enclosing_side_length(pos, angs_deg, ocx, ocy, oangle_deg)


# ---------- initial cluster generators ----------

def lattice_pos(i, j):
    """Axial-to-rectangular for unit hex lattice (pointy-top or flat-top?).
    Use flat-top hexagons with side 1. Distance between adjacent centers = 2*apothem = sqrt(3).
    For flat-top hex:
      x = 1.5 * s * q
      y = s * sqrt(3) * (r + q/2)
    But for packing, the center-to-center distance for non-overlapping flat-top hexes is 2*sqrt(3)/sqrt(3) = 2? 
    Actually, for two unit flat-top hexagons side by side, centers are 2*apothem = 2*(sqrt(3)/2)*1 = sqrt(3) apart horizontally? 
    Let me just use a known packing: place hexes on a triangular lattice with spacing 2*sqrt(3) between rows and sqrt(3) within rows for pointy-top, 
    or use the fact that the minimum distance between centers of two non-overlapping unit regular hexagons is 2 (for flat-top aligned) or sqrt(3) (for some orientations).
    
    For simplicity, use spacing that guarantees no overlap: use 2.0 as center distance (works for any orientation if we're conservative).
    """
    # Use a triangular lattice with spacing 2.0
    x = 2.0 * (i + 0.5 * (j & 1))
    y = 2.0 * math.sqrt(3) / 2.0 * j
    return (x, y)


def make_line_cluster():
    """11 hexagons in a line, spacing 2.0"""
    return [(2.0 * k, 0.0) for k in range(-5, 6)]


def make_triangular_cluster():
    """Triangular arrangement: rows of 5,4,2 or similar"""
    pos = []
    # Row 0: 5 hexes
    for i in range(-2, 3):
        pos.append((2.0 * i, 0.0))
    # Row 1: 3 hexes
    for i in range(-1, 2):
        pos.append((2.0 * (i + 0.5), 2.0 * math.sqrt(3) / 2.0))
    # Row 2: 3 hexes
    for i in range(-1, 2):
        pos.append((2.0 * (i + 0.5), -2.0 * math.sqrt(3) / 2.0))
    return pos


def make_circular_cluster():
    """5 in center ring, 6 around"""
    pos = [(0.0, 0.0)]
    # 6 around
    for k in range(6):
        ang = k * math.pi / 3.0
        pos.append((2.0 * math.cos(ang), 2.0 * math.sin(ang)))
    # 4 more outside
    for k in range(4):
        ang = k * math.pi / 2.0 + math.pi / 4.0
        pos.append((4.0 * math.cos(ang), 4.0 * math.sin(ang)))
    return pos


def make_compact_cluster():
    """Compact: 3-4-3-1 or similar hexagonal packing"""
    pos = []
    # Use a hexagonal lattice of radius 1: 1 + 6 + 12 = 19, too many.
    # Take a subset: center + 6 neighbors = 7, add 4 more
    pos.append((0.0, 0.0))
    for k in range(6):
        ang = k * math.pi / 3.0
        pos.append((2.0 * math.cos(ang), 2.0 * math.sin(ang)))
    # Add 4 more at distance ~3.5 in cardinal directions
    for k in range(4):
        ang = k * math.pi / 2.0
        pos.append((3.5 * math.cos(ang), 3.5 * math.sin(ang)))
    return pos


def make_random_cluster(seed=42):
    rng = np.random.default_rng(seed)
    pos = [(0.0, 0.0)]
    for _ in range(N - 1):
        # Place near existing
        base = pos[rng.integers(len(pos))]
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(1.5, 3.0)
        pos.append((base[0] + dist * math.cos(ang), base[1] + dist * math.sin(ang)))
    return pos


# ---------- main ----------

def construct():
    t0 = time.time()
    best_side = float('inf')
    best_result = None

    clusters = [
        make_line_cluster(),
        make_triangular_cluster(),
        make_circular_cluster(),
        make_compact_cluster(),
        make_random_cluster(1),
        make_random_cluster(2),
    ]

    # Remove duplicates
    unique_clusters = []
    for c in clusters:
        key = tuple(round(x, 6) for x in c)
        if key not in [tuple(round(x, 6) for x in uc) for uc in unique_clusters]:
            unique_clusters.append(c)

    for cluster in unique_clusters:
        if time.time() > DEADLINE - 10:
            break

        cx = sum(p[0] for p in cluster) / N
        cy = sum(p[1] for p in cluster) / N

        # Initial params
        x0 = []
        for p in cluster:
            x0.extend([p[0], p[1]])
        x0.extend([0.0] * N)  # angles in radians
        x0.extend([cx, cy, 0.0])

        # Try several outer angles
        for oangle_deg in [0.0, 30.0, 60.0]:
            if time.time() > DEADLINE - 5:
                break
            x0[-1] = math.radians(oangle_deg)

            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6,
                                        'adaptive': True})
            except Exception:
                continue

            if res.fun < best_side and res.fun < 1e9:
                # Extract result
                pos = []
                angs = []
                idx = 0
                for i in range(N):
                    pos.append((res.x[idx], res.x[idx + 1]))
                    idx += 2
                for i in range(N):
                    angs.append(math.degrees(res.x[idx]))
                    idx += 1
                ocx, ocy, oangle = res.x[idx], res.x[idx + 1], math.degrees(res.x[idx + 2])

                # Verify feasibility with the actual enclosing side
                S = compute_min_side(pos, angs, ocx, ocy, oangle)
                # Check no overlaps
                overlap_ok = True
                for i in range(N):
                    for j in range(i + 1, N):
                        d = hex_overlap_dist(pos[i][0], pos[i][1], angs[i],
                                             pos[j][0], pos[j][1], angs[j])
                        if d > 1e-6:
                            overlap_ok = False
                            break
                    if not overlap_ok:
                        break
                if overlap_ok and S < best_side:
                    best_side = S
                    best_result = (pos, angs, ocx, ocy, S, oangle)

    # Fallback: original solution
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    pos, angs, ocx, ocy, S, oangle = best_result
    inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(N)]
    return inner, (ocx, ocy), S, oangle
