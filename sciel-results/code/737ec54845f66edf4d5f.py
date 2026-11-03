"""Hexagon packing: lattice init + physics relaxation + minimal enclosing hexagon."""
import math
import time
import random
import numpy as np

N = 11
EPS = 1e-9
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# ---------- geometry helpers ----------
def hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (pointy-top at angle 0) centered at (cx,cy)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _ccw(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a, b, p):
    return (min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) - EPS and
            min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) - EPS)

def _seg_intersect(p1, p2, p3, p4):
    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and \
       ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)):
        return True
    if abs(d1) <= EPS and _on_segment(p3, p4, p1): return True
    if abs(d2) <= EPS and _on_segment(p3, p4, p2): return True
    if abs(d3) <= EPS and _on_segment(p1, p2, p3): return True
    if abs(d4) <= EPS and _on_segment(p1, p2, p4): return True
    return False

def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagons_disjoint(v1, v2, gap=1.001):
    """True if two hexagons (lists of 6 vertices) are disjoint (with small gap)."""
    # fast AABB reject
    x1min = min(p[0] for p in v1); x1max = max(p[0] for p in v1)
    y1min = min(p[1] for p in v1); y1max = max(p[1] for p in v1)
    x2min = min(p[0] for p in v2); x2max = max(p[0] for p in v2)
    y2min = min(p[1] for p in v2); y2max = max(p[1] for p in v2)
    if x1max + gap <= x2min or x2max + gap <= x1min or \
       y1max + gap <= y2min or y2max + gap <= y1min:
        return True
    # containment test
    if point_in_polygon(v1[0], v2):
        return False
    if point_in_polygon(v2[0], v1):
        return False
    # edge intersection
    n1 = len(v1); n2 = len(v2)
    for i in range(n1):
        a = v1[i]; b = v1[(i+1) % n1]
        for j in range(n2):
            c = v2[j]; d = v2[(j+1) % n2]
            if _seg_intersect(a, b, c, d):
                return False
    return True

# ---------- objective ----------
def compute_outer_side(positions, angles_deg):
    """Minimal side length of a regular hexagon (any center, any rotation) containing all unit hexagons."""
    best = float('inf')
    # sample outer rotations
    for adeg in range(0, 30, 3):
        a = math.radians(adeg)
        # support function: for each inner hexagon, max of dot(v - c, u) over its vertices
        # We want min over c of max_k sup_{v in hex_k} dot(v - c, u_k) where u_k are 6 normals
        # For a fixed outer orientation, the outer hexagon is intersection of 6 halfplanes.
        # Minimal enclosing regular hexagon with fixed orientation:
        # minimize R such that for all 6 directions u_i, max_k max_v dot(v, u_i) - R*dot(c,u_i) <= R
        # => R >= (max_k max_v dot(v,u_i) - R*dot(c,u_i)) for each i
        # This is a 2D LP-ish; solve by trying c on a grid or use the fact that optimal c
        # equalizes the 3 opposite pairs.
        # Simpler: the minimal enclosing hexagon (fixed orientation) centered at c has
        # R = max_i (max_k max_v dot(v, u_i) - dot(c, u_i)) = max_i (m_i - dot(c,u_i))
        # where m_i = max over all vertices of all hexes of dot(v, u_i).
        # minimize max_i (m_i - dot(c, u_i)) over c.
        # By symmetry of the 6 directions (u_{i+3} = -u_i), let m_i for i in 0..5.
        # objective f(c) = max_i (m_i - dot(c, u_i)).
        # This is convex in c. Solve via a few iterations of subgradient or grid.
        u = []
        for i in range(6):
            theta = a + i * math.pi / 3.0
            u.append((math.cos(theta), math.sin(theta)))
        # gather all vertices
        allv = []
        for (cx, cy), ad in zip(positions, angles_deg):
            for k in range(6):
                th = math.radians(ad) + k * math.pi / 3.0
                allv.append((cx + math.cos(th), cy + math.sin(th)))
        m = [0.0] * 6
        for v in allv:
            for i in range(6):
                d = v[0]*u[i][0] + v[1]*u[i][1]
                if d > m[i]:
                    m[i] = d
        # minimize max_i (m_i - dot(c,u_i)) over c=(cx,cy)
        # subgradient descent
        cx = 0.0; cy = 0.0
        lr = 0.5
        for it in range(200):
            # compute f and subgradient
            f = -1e9
            gx = 0.0; gy = 0.0
            active = []
            for i in range(6):
                val = m[i] - (cx*u[i][0] + cy*u[i][1])
                if val > f + 1e-6:
                    f = val
                    active = [i]
                elif abs(val - f) <= 1e-6:
                    active.append(i)
            for i in active:
                gx += -u[i][0]
                gy += -u[i][1]
            gn = math.hypot(gx, gy)
            if gn > 1e-12:
                cx -= lr * gx / gn
                cy -= lr * gy / gn
            lr *= 0.98
        # final value at c
        f = 0.0
        for i in range(6):
            val = m[i] - (cx*u[i][0] + cy*u[i][1])
            if val > f:
                f = val
        if f < best:
            best = f
    return best

def is_valid(positions, angles_deg, outer_center, outer_side, outer_angle):
    """Check all inner hexagons disjoint and inside outer."""
    inner_verts = []
    for (cx, cy), ad in zip(positions, angles_deg):
        inner_verts.append(hexagon_vertices(cx, cy, 1.0, ad))
    for i in range(len(inner_verts)):
        for j in range(i+1, len(inner_verts)):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                return False
    ov = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for iv in inner_verts:
        # all vertices of inner must be inside outer
        for v in iv:
            if not point_in_polygon(v, ov):
                return False
    return True

# ---------- lattice init ----------
def lattice_positions():
    """Hexagonal lattice positions for N unit hexagons (center of each)."""
    # hex lattice spacing: distance between centers of adjacent unit hexagons (edge to edge)
    # two unit hexagons sharing an edge: center distance = sqrt(3)
    s = math.sqrt(3.0)
    # generate hex lattice points, take N closest to origin
    pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = s * (q + r/2.0)
            y = s * (math.sqrt(3)/2.0) * r
            pts.append((x, y, q*q + r*r + q*r))
    pts.sort(key=lambda p: p[2])
    pos = [(p[0], p[1]) for p in pts[:N]]
    angles = [0.0] * N
    return pos, angles

# ---------- physics relaxation ----------
def relax(positions, angles_deg, iters=3000):
    """Repulsive + containment relaxation. Returns (positions, angles_deg)."""
    pos = [list(p) for p in positions]
    ang = [float(a) for a in angles_deg]
    n = len(pos)
    # estimate outer side from current
    outer_side = compute_outer_side(pos, ang)
    # containment: keep all inner vertices within outer_side*0.95 of center (0,0)
    # use radial spring
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        # compute forces
        fx = [0.0]*n; fy = [0.0]*n
        # repulsion between overlapping hexagons (use center distance heuristic for speed)
        min_dist = math.sqrt(3.0) - 0.01  # target center distance
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                d = math.hypot(dx, dy)
                if d < 1e-6:
                    d = 1e-6
                    dx = 1e-6
                if d < min_dist:
                    # push apart
                    f = (min_dist - d) * 2.0
                    fx[i] += dx/d * f
                    fy[i] += dy/d * f
                    fx[j] -= dx/d * f
                    fy[j] -= dy/d * f
        # containment: pull toward center if too far
        for i in range(n):
            r = math.hypot(pos[i][0], pos[i][1])
            max_r = outer_side * 0.85
            if r > max_r:
                f = (r - max_r) * 1.5
                fx[i] -= pos[i][0]/r * f
                fy[i] -= pos[i][1]/r * f
        # integrate
        for i in range(n):
            pos[i][0] += 0.1 * fx[i]
            pos[i][1] += 0.1 * fy[i]
    return [tuple(p) for p in pos], ang

# ---------- main ----------
def construct():
    t0 = time.time()
    pos, ang = lattice_positions()
    # relax
    pos, ang = relax(pos, ang, iters=2000)
    if time.time() > DEADLINE:
        side = compute_outer_side(pos, ang)
        return [[pos[i][0], pos[i][1], ang[i]] for i in range(N)], (0.0, 0.0), side, 0.0
    # compute outer side
    side = compute_outer_side(pos, ang)
    # second relaxation pass with tighter containment
    pos2, ang2 = relax(pos, ang, iters=1500)
    if time.time() > DEADLINE:
        side2 = compute_outer_side(pos2, ang2)
        return [[pos2[i][0], pos2[i][1], ang2[i]] for i in range(N)], (0.0, 0.0), side2, 0.0
    side2 = compute_outer_side(pos2, ang2)
    if side2 < side:
        pos, ang, side = pos2, ang2, side2
    return [[pos[i][0], pos[i][1], ang[i]] for i in range(N)], (0.0, 0.0), side, 0.0
