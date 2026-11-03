"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random

N = 12
TIME_LIMIT = 110.0  # safety margin below 120s

# Precompute hexagon geometry constants
def _hex_verts_local():
    # Unit hexagon (side 1), flat-top, vertices at 0,60,...,300 degrees
    return [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 60)]

_HEX_LOCAL = _hex_verts_local()
# Separation distance for two unit hexagons to be non-overlapping (disjoint)
_SEP = 2.0

def _transformed_hex(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + ca*x - sa*y, cy + sa*x + ca*y) for (x, y) in _HEX_LOCAL]

def _min_dist_poly(A, B):
    # Minimum distance between two convex polygons (vertex-to-edge + edge-to-edge)
    best = float('inf')
    def seg_dist(p, a, b):
        ax, ay = a; bx, by = b; px, py = p
        dx, dy = bx-ax, by-ay
        l2 = dx*dx + dy*dy
        if l2 == 0:
            return math.hypot(px-ax, py-ay)
        t = ((px-ax)*dx + (py-ay)*dy) / l2
        t = max(0.0, min(1.0, t))
        qx, qy = ax + t*dx, ay + t*dy
        return math.hypot(px-qx, py-qy)
    nA, nB = len(A), len(B)
    for i in range(nA):
        ax, ay = A[i]
        for j in range(nB):
            d = math.hypot(ax-B[j][0], ay-B[j][1])
            if d < best: best = d
        # edge i to vertex
        a2x, a2y = A[(i+1)%nA]
        for j in range(nB):
            d = seg_dist(B[j], (ax, ay), (a2x, a2y))
            if d < best: best = d
    return best

def _is_valid(polygons):
    # Check pairwise disjointness
    m = len(polygons)
    for i in range(m):
        for j in range(i+1, m):
            if _min_dist_poly(polygons[i], polygons[j]) < _SEP - 1e-7:
                return False
    return True

def _outer_side(polygons, center, angle_deg):
    # Minimum side length S of a regular hexagon (center, angle) containing all polygons
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Support function of a regular hexagon with side S: h(u) = S * (cos(u - a) for the 3 axes)
    # For a regular hexagon centered at origin with rotation a, the support in direction u is:
    # h(u) = S * max over the 3 normal directions of dot
    # The 3 outward normal directions (for flat-top with rotation a) are at angles a, a+120, a+240
    # Actually for a regular hexagon, support function h(u) = S * cos(theta) where theta is the
    # angle between u and the nearest vertex-normal... Let's use the standard form.
    # A regular hexagon with side S has apothem = S*sqrt(3)/2. The support in direction making
    # angle phi with the nearest normal is S*sqrt(3)/2 / cos(phi) for |phi|<=60.
    # Equivalently h(u) = S * f(angle) where f cycles with period 120.
    # Let's compute using the 6 edge normals.
    # Edge normals at angles: a + 90 + 60*k for k in 0..5? Let me just use the vertex form.
    # Vertices of outer hexagon (side S, rotation a, center c): c + S*(cos(a+60k), sin(a+60k))
    # A point p is inside iff for all 6 edges, the signed distance >= 0.
    # We need the smallest S such that all polygon points are inside.
    # For a given direction, the required S is determined by the max projection.
    # The regular hexagon (side S) is the intersection of 6 half-planes. Its support function:
    # h(u) = S * g(u) where g is the support of a unit-side hexagon.
    # For unit-side hexagon with vertex directions a+60k, the support in direction u:
    # g(u) = max_k dot(unit_vertex_dir, u) ... no, support of the polygon.
    # Simpler: the hexagon (side S, rotation a) = { x : for each of 6 outward normals n_k, n_k . (x-c) <= S*sqrt(3)/2 }
    # where n_k are unit normals at angles a+90+60k.
    # So for a point p, the required S from constraint k: S >= n_k . (p-c) / (sqrt(3)/2)
    # Overall S >= max over all points and all k of n_k . (p-c) / (sqrt(3)/2)
    # (and also the negative side, but by symmetry of hexagon we need both + and -; using all 6 normals covers it)
    apothem_unit = math.sqrt(3) / 2
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60*k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    max_need = 0.0
    for poly in polygons:
        for (px, py) in poly:
            for (nx, ny) in normals:
                val = (nx*(px-cx) + ny*(py-cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def construct():
    t0 = time.time()
    random.seed(12345)
    best = None
    best_score = float('inf')

    # Initial configuration: place hexagons in a rough grid, then optimize
    def make_init():
        # 3x4 grid arrangement, side 1 hexagons need ~2 spacing
        pts = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (0.866 if r % 2 else 0.0)
                y = r * 1.732
                pts.append([x - 3.0, y - 1.732, 0.0])
        return pts

    def score_config(cfg, center=(0.0, 0.0), angle=0.0):
        polys = [_transformed_hex(p[0], p[1], p[2]) for p in cfg]
        if not _is_valid(polys):
            return None
        return _outer_side(polys, center, angle)

    # Simulated annealing over inner positions + outer center + outer angle
    cfg = make_init()
    center = [0.0, 0.0]
    angle = 0.0
    cur_score = score_config(cfg, tuple(center), angle)
    if cur_score is None:
        # nudge to valid
        cur_score = 40.0
    best = (cfg[:], center[:], angle, cur_score)
    best_score = cur_score

    T = 2.0
    T_end = 0.01
    iters = 0
    while time.time() - t0 < TIME_LIMIT:
        iters += 1
        # pick a move
        r = random.random()
        if r < 0.6:
            idx = random.randrange(N)
            cfg[idx][0] += random.gauss(0, 0.3)
            cfg[idx][1] += random.gauss(0, 0.3)
        elif r < 0.8:
            idx = random.randrange(N)
            cfg[idx][2] += random.gauss(0, 10)
        elif r < 0.9:
            center[0] += random.gauss(0, 0.2)
            center[1] += random.gauss(0, 0.2)
        else:
            angle += random.gauss(0, 5)

        new_score = score_config(cfg, tuple(center), angle)
        if new_score is None:
            # invalid: penalize by reverting
            cfg[idx][0] -= random.gauss(0, 0.3) if r < 0.6 else 0
            # simpler: just reject
            continue
        if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = ([row[:] for row in cfg], center[:], angle, new_score)
        else:
            # revert
            if r < 0.6:
                cfg[idx][0] -= random.gauss(0, 0.3)
                cfg[idx][1] -= random.gauss(0, 0.3)
            elif r < 0.8:
                cfg[idx][2] -= random.gauss(0, 10)
            elif r < 0.9:
                center[0] -= random.gauss(0, 0.2)
                center[1] -= random.gauss(0, 0.2)
            else:
                angle -= random.gauss(0, 5)
        # cool
        T = max(T_end, T * 0.9995)

    bc, ba, bside = best[1][0], best[1][1], best[2]
    inner = [[p[0], p[1], p[2]] for p in best[0]]
    return inner, (bc, ba), bside, best[3] if best[3] is not None else bside

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
