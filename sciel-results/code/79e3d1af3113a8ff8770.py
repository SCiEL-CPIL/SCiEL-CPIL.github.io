"""Hexagon packing: minimize enclosing regular hexagon side for N unit hexagons."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
import numpy as np

def _hex_vertices(center, angle_deg, side=1.0):
    """Vertices of a regular hexagon with given center, rotation, side length."""
    cx, cy = center
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def _support_max(verts, directions):
    """Max projection of vertices onto given directions (array of unit vectors)."""
    # verts: (M,2), directions: (6,2)
    return np.max(verts @ directions.T, axis=0)

def _enclosing_side(points, angle_deg):
    """
    Minimal side length of a regular hexagon with the given rotation that
    encloses the set of points.
    A regular hexagon of side s centered at c, rotation a, has support in
    direction n_i (one of 6 normals) = s * cos(pi/6) = s*sqrt(3)/2  (inradius).
    Actually for a regular hexagon, the inradius (apothem) = s*sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes. For it to contain all
    points, for each normal direction n_i, max_i (p.n_i) - min_i(p.n_i) <= 2*apothem.
    So side s >= (max-min)/(2*apothem) for each i, take the max.
    """
    a = math.radians(angle_deg)
    # 6 normal directions of the hexagon (pointing to edge midpoints)
    # For a hexagon with vertices at angles a + i*pi/3, the edge normals are at a + pi/6 + i*pi/3
    normals = []
    for i in range(6):
        theta = a + math.pi / 6.0 + i * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    normals = np.array(normals)
    proj = points @ normals.T  # (P,6)
    widths = proj.max(axis=0) - proj.min(axis=0)  # (6,)
    apothem_factor = math.sqrt(3.0) / 2.0  # apothem = s * sqrt(3)/2
    # s = width / (2*apothem)
    sides = widths / (2.0 * apothem_factor)
    return float(np.max(sides))

def _cluster_positions(k):
    """
    Generate a compact hex-lattice cluster of unit hexagons.
    Returns list of (x, y, angle_deg) for N hexagons (padded/truncated).
    Honeycomb: horizontal spacing sqrt(3), vertical 1.5, odd rows offset sqrt(3)/2.
    """
    # We'll generate a hexagonal "ball" of hexes around origin in lattice coords.
    # Use axial coordinates (q, r) with cube constraint |q|,|r|,|q+r| <= k-1 for side k.
    # For N=11, side k=3 gives 19; we take the 11 closest to center.
    positions = []
    for k in range(1, 8):
        for q in range(-k, k + 1):
            for r in range(max(-k, -q - k), min(k, -q + k) + 1):
                # convert axial to pixel
                x = math.sqrt(3.0) * (q + r / 2.0)
                y = 1.5 * r
                dist = math.sqrt(x * x + y * y)
                positions.append((dist, x, y))
        if len(positions) >= N:
            break
    positions.sort(key=lambda t: t[0])
    out = []
    for dist, x, y in positions[:N]:
        out.append([x, y, 0.0])
    # If we somehow got fewer than N, pad with more lattice points
    if len(out) < N:
        allpos = []
        for k in range(1, 12):
            for q in range(-k, k + 1):
                for r in range(max(-k, -q - k), min(k, -q + k) + 1):
                    x = math.sqrt(3.0) * (q + r / 2.0)
                    y = 1.5 * r
                    allpos.append((x, y))
        allpos.sort(key=lambda t: t[0] * t[0] + t[1] * t[1])
        for x, y in allpos:
            if len(out) >= N:
                break
            if all(abs(x - o[0]) > 1e-6 or abs(y - o[1]) > 1e-6 for o in out):
                out.append([x, y, 0.0])
    return out[:N]

def _all_points(inner):
    """Collect all vertices of all inner hexagons."""
    pts = []
    for (x, y, ang) in inner:
        v = _hex_vertices((x, y), ang)
        pts.append(v)
    return np.vstack(pts)

def _best_rotation(inner, n_angles=360):
    """Find rotation of outer hexagon minimizing enclosing side."""
    points = _all_points(inner)
    best = (float('inf'), 0.0)
    for i in range(n_angles):
        ang = 360.0 * i / n_angles
        s = _enclosing_side(points, ang)
        if s < best[0]:
            best = (s, ang)
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    # Start with a compact hex-lattice cluster
    inner = _cluster_positions(N)
    best_side, best_angle = _best_rotation(inner, n_angles=720)
    best_inner = [list(r) for r in inner]

    # Local refinement: small random perturbations of inner positions + re-optimize rotation
    rng = np.random.default_rng(0)
    cur_inner = [list(r) for r in inner]
    cur_side, cur_angle = best_side, best_angle

    step = 0.05
    n_iter = 2000
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # propose a perturbation of one hexagon
        idx = int(rng.integers(0, N))
        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        da = rng.uniform(-10.0, 10.0)
        cand = [list(r) for r in cur_inner]
        cand[idx] = [cand[idx][0] + dx, cand[idx][1] + dy, cand[idx][2] + da]

        s, ang = _best_rotation(cand, n_angles=120)
        if s < cur_side - 1e-9:
            cur_inner = cand
            cur_side, cur_angle = s, ang
            if cur_side < best_side - 1e-9:
                best_side, best_angle = cur_side, cur_angle
                best_inner = [list(r) for r in cur_inner]
        # occasionally shrink step
        if it % 500 == 499:
            step *= 0.7

    # Final: re-optimize rotation finely for best_inner
    best_side, best_angle = _best_rotation(best_inner, n_angles=1440)
    best_inner = [list(r) for r in best_inner]

    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
