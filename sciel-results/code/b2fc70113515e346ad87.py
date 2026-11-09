"""Pack 11 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math

N = 11
_T0 = time.time()
_DEADLINE = 150.0

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _hex_vertices(cx, cy, a, ang_deg):
    """Vertices of a regular hexagon (circumradius a) centered at (cx,cy), rotated by ang_deg."""
    th = math.radians(ang_deg)
    pts = []
    for k in range(6):
        phi = th + k * math.pi / 3.0
        pts.append((cx + a * math.cos(phi), cy + a * math.sin(phi)))
    return pts

def _normals(ang_deg):
    """Unit outward normals of the 6 edges of a hexagon rotated by ang_deg (vertices at ang+k*60)."""
    th = math.radians(ang_deg)
    ns = []
    for k in range(6):
        # edge between vertex k and k+1 is perpendicular to direction th + (k+0.5)*60
        phi = th + (k + 0.5) * math.pi / 3.0
        ns.append((math.cos(phi), math.sin(phi)))
    return ns

def _required_side(vertices, center, outer_ang):
    """Smallest outer hexagon side (circumradius) of orientation outer_ang centered at
    `center` that contains all `vertices` (list of (x,y)). Returns side length or None."""
    ns = _normals(outer_ang)
    cx, cy = center
    m = 0.0
    for (x, y) in vertices:
        for (nx, ny) in ns:
            d = nx * (x - cx) + ny * (y - cy)
            if d > m:
                m = d
    apothem = m
    # apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return apothem / (math.sqrt(3.0) / 2.0)

def _all_vertices(inner, a=1.0):
    vs = []
    for (x, y, ang) in inner:
        vs.extend(_hex_vertices(x, y, a, ang))
    return vs

def _is_valid(inner, center, side, outer_ang, eps=1e-7):
    """Check every inner hexagon vertex is inside the outer hexagon (with eps margin)."""
    ns = _normals(outer_ang)
    cx, cy = center
    apothem = side * (math.sqrt(3.0) / 2.0) - eps
    vs = _all_vertices(inner)
    for (x, y) in vs:
        for (nx, ny) in ns:
            if nx * (x - cx) + ny * (y - cy) > apothem:
                return False
    return True

def _centroid(inner):
    sx = sy = 0.0
    for (x, y, a) in inner:
        sx += x; sy += y
    return (sx / len(inner), sy / len(inner))

def construct():
    # Honeycomb (triangular) center lattice of spacing sqrt(3) => non-overlapping unit hexagons.
    s3 = math.sqrt(3.0)
    u = (s3, 0.0)      # lattice vector 1
    v = (s3 / 2.0, 1.5)  # lattice vector 2 (60 deg)

    # Compact 11-point blob: a central hex + 6 neighbors + 4 axis extensions.
    ij = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # 7-point hexagon
        (2, 0), (-2, 0), (0, 2), (0, -2),  # 4 more
    ]
    assert len(ij) == N

    inner = []
    for (i, j) in ij:
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        inner.append([x, y, 0.0])

    # Try several outer orientations and centers; keep the smallest valid side.
    best = None
    cent = _centroid(inner)
    candidate_centers = [cent, (0.0, 0.0)]
    for outer_ang in (0.0, 15.0, 30.0, 45.0, 7.5, 22.5):
        for center in candidate_centers:
            side = _required_side(_all_vertices(inner), center, outer_ang)
            if side is None:
                continue
            # safety margin against numerical/evaluator tolerance
            side_safe = side * 1.00005 + 1e-6
            if _is_valid(inner, center, side_safe, outer_ang):
                if best is None or side_safe < best[2]:
                    best = (inner, center, side_safe, outer_ang)
        if _time_left() < 5.0:
            break

    if best is None:
        # Fallback: a guaranteed (if large) valid configuration.
        inner = [[s3 * 2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_out, center_out, side_out, ang_out = best
    # Return inner as list of [x, y, angle_degrees]
    return inner_out, (center_out[0], center_out[1]), side_out, ang_out
