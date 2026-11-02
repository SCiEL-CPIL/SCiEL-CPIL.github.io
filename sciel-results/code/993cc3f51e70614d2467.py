import numpy as np
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    R = side
    inradius = R * math.sqrt(3) / 2
    for i in range(6):
        theta = i * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        if rx * nx + ry * ny > inradius + 1e-9:
            return False
    return True

def all_points_in_hexagon(points, cx, cy, side, angle_deg=0.0):
    """Check if all points are inside the hexagon."""
    for px, py in points:
        if not point_in_hexagon(px, py, cx, cy, side, angle_deg):
            return False
    return True

def min_side_to_contain(points, cx, cy, angle_deg=0.0):
    """Find minimal side length of hexagon centered at (cx,cy) with given rotation that contains all points."""
    lo = 0.0
    hi = 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if all_points_in_hexagon(points, cx, cy, mid, angle_deg):
            hi = mid
        else:
            lo = mid
    return hi

def compute_best_packing(n):
    """Generate hexagonal lattice packing of n unit hexagons, find minimal containing hexagon."""
    k = 1
    while 3 * k * (k - 1) + 1 < n:
        k += 1
    
    positions = []
    for q in range(-(k-1), k):
        for r in range(-(k-1), k):
            if abs(q + r) > k - 1:
                continue
            positions.append((q, r))
    
    positions.sort(key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
    positions = positions[:n]
    
    sqrt3 = math.sqrt(3)
    centers = []
    for q, r in positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    centers = [(x - avg_x, y - avg_y) for x, y in centers]
    
    all_vertices = []
    for cx, cy in centers:
        verts = hexagon_vertices(cx, cy, 1.0, 0.0)
        all_vertices.extend(verts)
    
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in range(0, 30, 1):
        side = min_side_to_contain(all_vertices, 0.0, 0.0, float(angle_deg))
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    for angle_deg in np.arange(max(0, best_angle - 2), min(30, best_angle + 2), 0.1):
        side = min_side_to_contain(all_vertices, 0.0, 0.0, float(angle_deg))
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return centers, best_side, best_angle

# EVOLVE-BLOCK-START
def construct():
    centers, side, angle = compute_best_packing(N)
    inner = [[x, y, 0.0] for x, y in centers]
    return inner, (0.0, 0.0), side, angle
# EVOLVE-BLOCK-END
