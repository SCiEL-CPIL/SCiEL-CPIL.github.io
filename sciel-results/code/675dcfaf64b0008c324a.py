"""Packing 11 unit hexagons inside a regular hexagon - flower/cluster design."""
import math
import time

N = 11
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon (flat-top or pointy per rotation)."""
    th = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = th + math.radians(60.0 * k)
        pts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return pts


def _support(poly, dx, dy):
    """Max of dot((p - c), d) over polygon vertices; c is polygon center."""
    m = -1e18
    for (px, py) in poly:
        v = (px - poly[0][0]) * 0 + (px) * dx + (py) * dy
        # support relative to origin; we'll shift later
    # We need support relative to the *cluster centroid*; compute raw max then subtract centroid dot.
    return m


def _cluster_support_value(centers, radii, dx, dy):
    """For each inner hex (center + circumradius r), the support in direction (dx,dy) is
    center dot d + r. The cluster support = max over inner hexes of (c_i . d + r_i).
    The minimal enclosing regular hexagon (same orientation) side = max over 6 directions of
    (support_+ + support_-)/2 where support_- = -min(c_i . d - r_i) = max(-c_i.d + r_i).
    Actually for a regular hexagon container of side s centered at origin with orientation phi,
    the half-width in the 6 face-normal directions is s/2 * ... Let's use: container hexagon
    with circumradius R=s has apothem = s*sqrt(3)/2. A point (x,y) is inside iff for all 6
    face-normals n_k: dot(p, n_k) <= apothem. So we need max_k max_i (dot(c_i, n_k) + r_i)
    to be <= apothem = s*sqrt(3)/2. Hence s >= (2/sqrt(3)) * max_k max_i (dot(c_i,n_k)+r_i).
    Also must handle the opposite face: dot(c_i, -n_k) + r_i <= apothem -> same since we take
    max over all 6 normals including opposites. So s_min = (2/sqrt(3)) * M where
    M = max over 6 face-normals n_k of max_i (dot(c_i, n_k) + r_i), with r_i = 1 (unit hex circumradius).
    """
    M = 0.0
    for k in range(6):
        a = math.radians(60.0 * k)  # face normals at 0,60,...
        nx, ny = math.cos(a), math.sin(a)
        for (cx, cy) in centers:
            v = cx * nx + cy * ny + 1.0
            if v > M:
                M = v
    return (2.0 / math.sqrt(3.0)) * M


def _side_for_centers(centers):
    """Minimal outer side (same orientation, centered at origin) for given centers."""
    return _cluster_support_value(centers, 0.0, 0.0)


def _translate_to_center(centers):
    """Translate centers so their 'support center' is at origin (balance)."""
    # Compute the centroid that minimizes max support: use the average of extremes.
    # Simple: translate by the average of all centers.
    sx = sum(c[0] for c in centers) / len(centers)
    sy = sum(c[1] for c in centers) / len(centers)
    return [(c[0] - sx, c[1] - sy) for c in centers]


def _build_flower11():
    """
    Build 11 unit-hexagon centers in a cluster that packs tightly.
    Use axial coordinates for a hex grid, convert to Cartesian.
    Layout: a 3-4-4 row pattern in axial coords (rows of hexes).
    Axial (q, r) -> Cartesian (x = 1.5*q, y = sqrt(3)*(r + q/2)) for pointy-top.
    But our hexagons are unit side; distance between adjacent centers = sqrt(3) (apothem*2)
    for pointy-top? For a regular hex of side 1, the distance between centers of two
    edge-sharing hexes is 2*apothem = sqrt(3). Let's use that spacing.
    """
    # Use axial coords with spacing: x = 1.5*q, y = sqrt(3)*(r + 0.5*q)
    # This gives neighbor distance sqrt(3). Good (edge-to-edge).
    # 11 cells: a compact cluster. Use these axial coords (a 3-4-4 arrangement):
    axial = [
        (0, 0), (1, 0), (2, 0),        # row r=0: 3 cells
        (-1, 1), (0, 1), (1, 1), (2, 1),  # row r=1: 4 cells
        (-1, 2), (0, 2), (1, 2), (2, 2),  # row r=2: 4 cells
    ]
    sq3 = math.sqrt(3.0)
    centers = []
    for (q, r) in axial:
        x = 1.5 * q
        y = sq3 * (r + 0.5 * q)
        centers.append((x, y))
    return centers


def construct():
    global _START
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin

    # Primary design: 3-4-4 axial cluster
    centers = _build_flower11()
    centers = _translate_to_center(centers)
    side = _side_for_centers(centers)

    # Try a few rotations of the inner cluster to reduce the outer side.
    best_side = side
    best_centers = centers
    for rot_deg in range(0, 180, 15):
        th = math.radians(rot_deg)
        crot = []
        for (cx, cy) in centers:
            x = cx * math.cos(th) - cy * math.sin(th)
            y = cx * math.sin(th) + cy * math.cos(th)
            crot.append((x, y))
        crot = _translate_to_center(crot)
        s = _side_for_centers(crot)
        if s < best_side:
            best_side = s
            best_centers = crot
        if time.time() > deadline:
            break

    # Build inner rows: (x, y, angle_degrees). Inner hexagons are unit side, orientation 0.
    inner = []
    for (cx, cy) in best_centers:
        inner.append([float(cx), float(cy), 0.0])

    # Outer hexagon: center at origin, side = best_side, rotation 0.
    # Pad a tiny epsilon to ensure strict containment numerically.
    outer_side = best_side + 1e-9
    return inner, (0.0, 0.0), float(outer_side), 0.0


# Self-test (not called by evaluator)
if __name__ == "__main__":
    res = construct()
    print("inner count:", len(res[0]))
    print("outer side:", res[2])
