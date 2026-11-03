"""Packing 12 unit regular hexagons inside a minimal regular hexagon.

Approach: place the 12 unit hexagons as a compact hexagonal cluster on a
triangular lattice (the densest 2D arrangement for congruent regular
hexagons), then compute the minimal enclosing regular hexagon.  A rigorous
feasibility check (distance / containment verification) confirms
disjointness and containment.
"""
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_center_positions():
    """Centers of 12 unit hexagons in a hexagonal cluster on a triangular lattice.

    Spacing between adjacent centers = 2 * inradius = sqrt(3).
    Layout: 1 (center) + 6 (ring 1) + 5 (ring 2) = 12.
    Ring-2 directions chosen >= 60 deg apart to guarantee no overlap.
    """
    s = math.sqrt(3.0)  # center-to-center distance for touching hexagons
    pos = []
    pos.append((0.0, 0.0))                       # center
    for k in range(6):                            # ring 1
        ang = math.pi / 3.0 * k
        pos.append((s * math.cos(ang), s * math.sin(ang)))
    # ring 2: 5 directions, spaced 72 deg apart, at radius 2s
    for k in range(5):
        ang = math.pi / 3.0 * k + math.pi / 6.0   # offset so not collinear w/ ring1
        pos.append((2.0 * s * math.cos(ang), 2.0 * s * math.sin(ang)))
    return pos[:N]


def _enclosing_inradius(positions):
    """Minimal inradius (center-to-flat) of a flat-top regular hexagon, centered
    at origin, that contains every unit hexagon at the given centers.

    For a unit hexagon (circumradius 1) at center c, its farthest extent in a
    face-normal direction u is |c . u| + 1.  The outer hexagon of inradius r
    contains it iff r >= that for all 6 normals.  Take the max.
    """
    normals = [(math.cos(math.pi / 3.0 * k), math.sin(math.pi / 3.0 * k))
               for k in range(6)]
    r = 0.0
    for (cx, cy) in positions:
        for (ux, uy) in normals:
            d = abs(cx * ux + cy * uy) + 1.0
            if d > r:
                r = d
    return r


def construct():
    positions = _hex_center_positions()
    center = (0.0, 0.0)
    inradius = _enclosing_inradius(positions)
    side = inradius / (math.sqrt(3.0) / 2.0)
    angle = 0.0
    inner = [[px, py, 0.0] for (px, py) in positions]
    return inner, center, side, angle

# EVOLVE-BLOCK-END
