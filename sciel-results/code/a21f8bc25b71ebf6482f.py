import time
import math
import numpy as np

N = 11

# ---- Geometry helpers -------------------------------------------------------

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (flat-top when angle=0) centered at (cx,cy).
    angle_deg rotates the hexagon about its center.
    For a unit hexagon, the vertices at angle 0 (flat-top) are at angles
    0,60,...,300 degrees with radius = side (circumradius = side).
    """
    a = math.radians(angle_deg)
    r = side  # circumradius of a regular hexagon equals its side length
    pts = []
    for k in range(6):
        th = a + math.radians(60 * k)
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def half_width_at_angle(cx, cy, side, angle_deg, theta):
    """Half-extent of a regular hexagon (flat-top at angle 0, rotated by
    angle_deg) along direction theta.  For a regular hexagon of circumradius
    r, the support function in direction theta is r * max over the 6 vertex
    normals.  For a regular hexagon the support function is:
        h(theta) = r * max_{k} cos(theta - (angle + 60k))  but only where the
        corresponding face/vertex is active.
    A simpler exact formula: the support function of a regular hexagon with
    circumradius r and orientation phi is
        h(theta) = r * max( cos(theta-phi), cos(theta-phi-60), cos(theta-phi-120),
                            cos(theta-phi+60), cos(theta-phi+120), cos(theta-phi+180) )
    but capped by the apothem in the face directions.  Actually the exact support
    function of a regular hexagon is piecewise; the maximum of the 6 vertex
    dot products gives the support in the vertex directions, and between
    vertices the support is the apothem * secant.  The cleanest exact form:
        h(theta) = apothem / cos( (theta - phi) mod 60 - 30 )   when that angle
        is within [-30,30], where apothem = r * cos(30) = r * sqrt(3)/2.
    This is exact for a regular hexagon.
    """
    apothem = side * math.sqrt(3.0) / 2.0
    phi = math.radians(angle_deg)
    d = (theta - phi) % (math.pi / 3.0) - math.pi / 6.0   # in [-30deg,30deg]
    return apothem / math.cos(d)


def max_extent_of_points(points, angle_deg):
    """For a set of points, return the minimal side length S of a regular
    hexagon (flat-top orientation = angle_deg, centered at origin) that
    contains all points.  The hexagon centered at origin with orientation
    angle_deg and side S contains a point p iff for every direction theta
    (6 face normals), the projection |p . n| <= h(theta) where h is the
    support function of the hexagon in that face-normal direction.
    The 6 face normals are at angles angle_deg + 30 + 60k.
    The support in a face-normal direction equals the apothem = S*sqrt(3)/2.
    So containment iff for all 6 face normals n_k: |p . n_k| <= S*sqrt(3)/2.
    Hence S >= (2/sqrt(3)) * max_k max_p |p . n_k|.
    """
    phi = math.radians(angle_deg)
    maxproj = 0.0
    for k in range(6):
        th = phi + math.radians(30 + 60 * k)
        nx, ny = math.cos(th), math.sin(th)
        for (x, y) in points:
            v = abs(x * nx + y * ny)
            if v > maxproj:
                maxproj = v
    return (2.0 / math.sqrt(3.0)) * maxproj


def cluster_points(angle_deg):
    """Return the (x,y) centers of the 11-hexagon centered cluster, rotated
    by angle_deg about the cluster centroid (origin)."""
    # Centers in axial-like layout for a centered hex cluster of 11:
    # ring0: (0,0)
    # ring1: 6 hexagons at distance 2 (flat-to-flat of unit hex = sqrt(3))
    # ring2: 4 hexagons (the "corners" of the ring-2, i.e. at 2*sqrt(3) at
    #        0,120,240 and ... actually ring2 has 12 positions; we take the
    #        4 that are at the 60-degree-spaced "axis" directions? No.)
    # For a centered hexagonal number 11 = 1 + 6 + 4, the second ring is
    # incomplete: it has 4 hexagons.  The natural completion of the 11-cluster
    # is the 1 (center) + 6 (first ring) + 4 (second ring at the 0,120,240
    # and 180? ).  Actually the standard "11 hexagons" is a hexagon of side 2
    # minus 5?  Let me use the known optimal: a 11-hex cluster = center + 6 +
    # 4 where the 4 are at the four "edge-mid" directions of ring 2.
    # I'll instead build it as: all lattice points (in the hex grid) with
    # hex-distance <= 1 (7 hexes) plus the 4 ring-2 points at angles 0,120,240,
    # and ... we need 4 more.  The 4 ring-2 points that keep the cluster
    # symmetric under 3-fold are at 0,120,240 (3 points) -- that's only 3.
    # 11 = 1+6+4, so 4 ring-2 points.  For 3-fold symmetry we'd need a multiple
    # of 3, so 4 breaks 3-fold symmetry.  The known optimal 11-packing uses a
    # 1+6+4 arrangement that is symmetric under reflection only.
    #
    # Simplest robust approach: place 11 hexes as a "flower" of 7 (center+ring1)
    # plus 4 more attached to alternating ring-1 hexes to make a compact shape.
    # I'll generate ring-2 positions and pick 4 that minimize the bounding hex.

    # Hex grid: use pointy layout.  Unit hex, center-to-center distance d=sqrt(3).
    d = math.sqrt(3.0)
    pts = [(0.0, 0.0)]
    # ring 1: 6 neighbors
    for k in range(6):
        th = math.radians(60 * k)
        pts.append((d * math.cos(th), d * math.sin(th)))
    # ring 2: 12 positions; we will choose 4.
    ring2 = []
    for k in range(12):
        th = math.radians(30 * k)
        ring2.append((2 * d * math.cos(th), 2 * d * math.sin(th)))
    # Choose 4 ring-2 points.  For the known optimal 11-cluster, the 4 are the
    # ones at angles 0, 120, 240 and 180?  That set {0,120,180,240} is not
    # symmetric.  Instead try all 4-subsets?  12 choose 4 = 495, cheap.
    # We'll pick the subset that minimizes max_extent at angle 0.
    best_subset = None
    best_val = float('inf')
    from itertools import combinations
    for sub in combinations(range(12), 4):
        cand = pts + [ring2[i] for i in sub]
        v = max_extent_of_points(cand, 0.0)
        if v < best_val:
            best_val = v
            best_subset = sub
    pts = pts + [ring2[i] for i in best_subset]

    # rotate by angle_deg
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for (x, y) in pts:
        out.append((x * ca - y * sa, x * sa + y * ca))
    return out


def best_orientation():
    """Search over the 6 discrete orientations (multiples of 30 deg) for the
    minimal enclosing hexagon side length, and also try translating the outer
    hexagon to the centroid of the cluster (which for a symmetric cluster is
    the origin; we shift points so the centroid is at origin before measuring,
    which is equivalent to centering the outer hexagon on the centroid)."""
    best_side = float('inf')
    best_ang = 0.0
    for ang in range(0, 360, 30):
        pts = cluster_points(ang)
        # center points on their centroid
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts = [(x - cx, y - cy) for (x, y) in pts]
        s = max_extent_of_points(pts, 0.0)
        if s < best_side:
            best_side = s
            best_ang = ang
    return best_side, best_ang


# ---- Construct ------------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 120.0   # safety margin under the 150s wall-clock kill

    side, ang = best_orientation()

    # Build the inner list: 11 rows (x, y, angle_degrees).
    # All inner hexagons share the cluster orientation 'ang'.
    pts = cluster_points(ang)
    inner = [[round(x, 9), round(y, 9), float(ang)] for (x, y) in pts]

    # Outer hexagon centered at the cluster centroid (origin here since we
    # built the cluster about the origin), side = best_side, rotation = 0
    # (flat-top) which matches the orientation used in max_extent.
    center = (0.0, 0.0)
    outer_side = float(side)
    outer_angle = 0.0

    # Quick validity sanity: ensure we have exactly 11 inner hexagons.
    if len(inner) != N:
        # Fallback to a safe (larger) configuration.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Deadline guard: if we are dangerously close, return immediately.
    if time.time() > deadline - 5.0:
        return inner, center, outer_side, outer_angle

    return inner, center, outer_side, outer_angle


if __name__ == "__main__":
    inner, center, side, angle = construct()
    print("inner count:", len(inner))
    print("outer side:", side)
    print("outer angle:", angle)
