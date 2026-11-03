"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
import math
import time as _time

def _hex_corner_pts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (flat-top when angle=0)."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        pts.append((cx + side * math.cos(th), cy + side * math.sin(th)))
    return pts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Return (max_overall, max_for_this_point) of signed distance to hex boundary.
    Positive => inside. For a convex polygon, signed distance to boundary = min over edges."""
    a = math.radians(angle_deg)
    # edge normals (outward) for a regular hexagon
    # edge k is between vertex k and vertex k+1
    # We compute for each edge the signed distance (positive inside)
    best = float('inf')  # min signed distance over edges (closest edge)
    for k in range(6):
        th1 = a + k * math.pi / 3.0
        th2 = a + (k + 1) * math.pi / 3.0
        v1 = (cx + side * math.cos(th1), cy + side * math.sin(th1))
        v2 = (cx + side * math.cos(th2), cy + side * math.sin(th2))
        # edge vector
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        L = math.hypot(ex, ey)
        # outward normal: for a CCW polygon, outward normal of edge (e) is (ey, -ex)/L
        nx = ey / L
        ny = -ex / L
        # signed distance from point to this edge line, positive if on inside (toward center)
        # inside means the center is on the negative side of outward normal
        d = (px - v1[0]) * nx + (py - v1[1]) * ny
        # inside => d <= 0 ; signed distance to boundary (positive inside) = -d
        sd = -d
        if sd < best:
            best = sd
    return best

def _inner_hex_corners(x, y, angle_deg):
    """6 corners of a unit hexagon at (x,y) with rotation angle_deg (flat-top at 0)."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        pts.append((x + math.cos(th), y + math.sin(th)))
    return pts

def _hex_side_length_for_cluster(s):
    """For a hexagonal cluster of side s (rows of 1..s..1), the bounding hexagon side length
    when aligned. This is 2*s - 1 in terms of... let's compute the actual span.
    Actually for hexagonal lattice with spacing sqrt(3) in x and 1.5 in y, a cluster of side s
    spans from -x_max to x_max. We'll just rely on optimization."""
    return 2.0 * s - 1.0

def _build_cluster(s):
    """Build hexagonal cluster of side s (number of hexagons per side).
    Returns list of (x, y) centers. Hexagons are flat-top, angle 0.
    Axial coordinates: for a hexagon cluster of side s, centers at
    q in [-(s-1)..s-1], r in [max(-(s-1), -q-(s-1)) .. min(s-1, -q+(s-1))]
    Convert axial to pixel for flat-top:
      x = spacing_x * (q + r/2)
      y = spacing_y * r
    For unit hexagons, flat-top, the distance between adjacent centers horizontally is sqrt(3),
    and the vertical spacing between rows is 1.5.
    """
    centers = []
    for q in range(-(s - 1), s):
        rmin = max(-(s - 1), -q - (s - 1))
        rmax = min(s - 1, -q + (s - 1))
        for r in range(rmin, rmax + 1):
            x = math.sqrt(3.0) * (q + r / 2.0)
            y = 1.5 * r
            centers.append((x, y))
    return centers

def _objective(params, centers, angle_inner):
    """params = [cx, cy, side, angle_deg_outer]. Minimize side subject to all inner corners inside."""
    cx, cy, side, ang = params
    if side < 0.5:
        return 1e6 + side
    min_sd = float('inf')
    for (x, y) in centers:
        corners = _inner_hex_corners(x, y, angle_inner)
        for (px, py) in corners:
            sd = _point_in_hex(px, py, cx, cy, side, ang)
            if sd < min_sd:
                min_sd = sd
    # penalty: want min_sd >= 0. Use a soft penalty.
    # We minimize side, but must keep min_sd >= 0.
    # Use: side + big_penalty * max(0, -min_sd)
    pen = max(0.0, -min_sd)
    return side + 1000.0 * pen

def _build_and_optimize(s, angle_inner, time_budget):
    centers = _build_cluster(s)
    # initial guess
    # compute bounding box of centers to estimate side
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    xmid = (max(xs) + min(xs)) / 2.0
    ymid = (max(ys) + min(ys)) / 2.0
    # rough side: the cluster spans about 2*(s-1)*sqrt(3)/... let's estimate
    # The distance from center to farthest center ~ (s-1)*sqrt(3)
    # plus 1 for the hexagon radius. Side length of enclosing hexagon ~ that + 1
    rough_side = (s - 1) * math.sqrt(3.0) + 1.5
    x0 = [xmid, ymid, rough_side, 0.0]
    
    best_side = rough_side
    best_params = x0[:]
    
    start = _time.time()
    # Nelder-Mead
    try:
        res = minimize(_objective, x0, args=(centers, angle_inner),
                       method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-5,
                                'adaptive': True})
        if res.success or res.fun < _objective(best_params, centers, angle_inner):
            # verify feasibility
            cx, cy, side, ang = res.x
            min_sd = float('inf')
            for (x, y) in centers:
                corners = _inner_hex_corners(x, y, angle_inner)
                for (px, py) in corners:
                    sd = _point_in_hex(px, py, cx, cy, side, ang)
                    if sd < min_sd:
                        min_sd = sd
            if min_sd >= -1e-6:
                best_side = side
                best_params = res.x[:]
    except Exception:
        pass
    
    elapsed = _time.time() - start
    return best_side, best_params, centers, elapsed

def construct():
    deadline = 150.0 - 10.0  # safety margin
    start = _time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Try different cluster sizes and inner hexagon rotations
    # The cluster side s determines number of hexagons. We want a good balance.
    # Start with s that gives a reasonable count.
    # For s=1: 1 hexagon, side ~ 1
    # For s=2: 7 hexagons
    # For s=3: 19 hexagons
    # For s=4: 37 hexagons
    # For s=5: 61 hexagons
    # We'll try s from 1 up to a budget-limited max.
    
    # Inner hexagon rotation: 0 (flat-top, aligned with lattice) is natural.
    # But we can also try 30 degrees (pointy-top) which might fit differently.
    # Actually the lattice is built for flat-top (angle 0). Let's keep angle_inner=0.
    
    angle_inner = 0.0
    
    # Try increasing s until time runs low
    s = 1
    while s <= 12:
        if _time.time() > deadline:
            break
        side, params, centers, elapsed = _build_and_optimize(s, angle_inner, deadline)
        if side < best_side:
            best_side = side
            # Build inner list
            inner = [[c[0], c[1], angle_inner] for c in centers]
            cx, cy, side_val, ang = params
            best_result = (inner, (cx, cy), side_val, ang)
        s += 1
    
    if best_result is None:
        # Fallback: single hexagon
        inner = [[0.0, 0.0, 0.0]]
        return inner, (0.0, 0.0), 1.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
