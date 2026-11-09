"""Packing 12 unit regular hexagons into a minimal regular hexagon container."""
import math
import time
import random
import numpy as np

N = 12
# Deadline guard: evaluator kills at 150s, keep safety margin
DEADLINE = 135.0

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def segment_intersects_segment(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (including endpoints)."""
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1]):
        return True
    if o2 == 0 and min(p1[0], p2[0]) <= p4[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p4[1] <= max(p1[1], p2[1]):
        return True
    if o3 == 0 and min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= p1[1] <= max(p3[1], p4[1]):
        return True
    if o4 == 0 and min(p3[0], p4[0]) <= p2[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= p2[1] <= max(p3[1], p4[1]):
        return True
    
    return False

def point_in_polygon(pt, polygon):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_overlap(v1, v2):
    """Check if two convex hexagons overlap using SAT or edge intersection + containment."""
    # Check if any edges intersect
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        for j in range(n2):
            if segment_intersects_segment(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return True
    # Check if one is contained in the other
    if point_in_polygon(v1[0], v2):
        return True
    if point_in_polygon(v2[0], v1):
        return True
    return False

def get_inner_vertices(positions, angles):
    """Get all 12 hexagon vertex lists."""
    verts = []
    for i in range(N):
        verts.append(hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    return verts

def check_inner_collisions(verts):
    """Check if any two inner hexagons overlap."""
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap(verts[i], verts[j]):
                return True
    return False

def check_outer_containment(verts, outer_verts):
    """Check if all inner hexagons are inside the outer hexagon."""
    for i in range(N):
        for v in verts[i]:
            if not point_in_polygon(v, outer_verts):
                return False
    return True

def compute_side_length(positions, angles, outer_angle_deg):
    """Compute the minimum side length of a regular hexagon (with given rotation) 
    centered at origin that contains all inner hexagons."""
    # For a regular hexagon with side s and rotation angle, we need to find
    # the minimum s such that all vertices of inner hexagons are inside.
    # A regular hexagon centered at origin with side s and rotation phi has
    # support function: for direction theta, the distance from center to boundary
    # is s * cos(phi - theta) / cos(pi/6) for the relevant face... 
    # Actually, let's use the property that a regular hexagon with side s centered at origin
    # and rotated by angle phi contains point (x,y) iff for all 6 face normals,
    # the dot product is <= s * cos(pi/6).
    # The face normals are at angles phi + k*pi/3 for k=0..5, and the distance
    # from center to each face is s * cos(pi/6) = s * sqrt(3)/2.
    
    s_min = 0.0
    sqrt3_over_2 = math.sqrt(3.0) / 2.0
    phi = math.radians(outer_angle_deg)
    
    for i in range(N):
        for j in range(6):
            vx, vy = positions[i][0] + math.cos(math.radians(angles[i]) + j * math.pi/3.0)
            vy = positions[i][1] + math.sin(math.radians(angles[i]) + j * math.pi/3.0)
            for k in range(6):
                normal_angle = phi + k * math.pi / 3.0
                nx, ny = math.cos(normal_angle), math.sin(normal_angle)
                dist = abs(vx * nx + vy * ny)
                # For the hexagon to contain the point, dist <= s * sqrt(3)/2
                # So s >= dist / (sqrt(3)/2)
                required_s = dist / sqrt3_over_2
                if required_s > s_min:
                    s_min = required_s
    return s_min

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = 40.0
    best_positions = [[4.0 * k, 0.0] for k in range(-(N//2), N - N//2)]
    best_angles = [0.0] * N
    best_outer_angle = 0.0
    
    # Initial setup: place hexagons in a honeycomb-like pattern
    # A regular hexagon of side 1 has width 2 (flat-to-flat) and height sqrt(3) (point-to-point)
    # Let's try a hexagonal arrangement of 12 hexagons
    
    # Honeycomb: center + 6 around + 5 more
    # Actually, let's use a known dense packing arrangement
    # 12 hexagons can be arranged as: 1 center, 6 in first ring, 5 in second ring (partial)
    # Or: rows of 3,4,3,2 or similar
    
    # Let's try: two rows of 6, offset
    # Or a 3-4-3-2 arrangement
    
    # For now, use a simple hexagonal grid arrangement
    # Hex grid with spacing: horizontal = 3 (2*1.5), vertical = sqrt(3)
    # Actually for flat-top hexagons side 1:
    # Horizontal spacing: 3 (if flat-to-flat is 2, but we need gap)
    # For touching: horizontal dist = 3, vertical dist = sqrt(3)
    
    # Let's place them in a compact hexagonal pattern
    # Ring 0: 1 hex at center
    # Ring 1: 6 hexes around
    # Ring 2: partial (5 more to make 12)
    
    # Positions for hexagonal grid (flat-top, side 1)
    # Center hex at (0,0)
    # Ring 1: at distance 3 (center-to-center for touching flat-top hexes)
    #   angles: 0, 60, 120, 180, 240, 300
    # Ring 2: at distance 6, but we only need 5
    
    r1 = 3.0  # center-to-center distance for touching flat-top hexagons
    r2 = 6.0
    
    pos = [(0.0, 0.0)]
    for k in range(6):
        angle = k * math.pi / 3.0
        pos.append((r1 * math.cos(angle), r1 * math.sin(angle)))
    # Add 5 more in ring 2
    for k in range(5):
        angle = k * 2 * math.pi / 5.0
        pos.append((r2 * math.cos(angle), r2 * math.sin(angle)))
    
    initial_positions = [list(p) for p in pos]
    initial_angles = [0.0] * N
    
    # Simulated annealing
    T = 5.0
    T_min = 0.001
    cooling = 0.999
    max_iters = 200000
    
    positions = [p[:] for p in initial_positions]
    angles = initial_angles[:]
    
    # Precompute initial state
    verts = get_inner_vertices(positions, angles)
    inner_ok = not check_inner_collisions(verts)
    side = compute_side_length(positions, angles, 0.0)
    
    if inner_ok and side < best_side:
        best_side = side
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE:
            break
        if T < T_min:
            break
        
        # Pick a random hexagon to move
        idx = random.randint(0, N - 1)
        old_px, old_py = positions[idx]
        old_angle = angles[idx]
        
        # Random perturbation
        dx = random.gauss(0, 0.5)
        dy = random.gauss(0, 0.5)
        da = random.gauss(0, 10.0)
        
        positions[idx] = [old_px + dx, old_py + dy]
        angles[idx] = old_angle + da
        
        # Check validity
        verts = get_inner_vertices(positions, angles)
        inner_ok = not check_inner_collisions(verts)
        
        if inner_ok:
            new_side = compute_side_length(positions, angles, 0.0)
        else:
            new_side = side + 100.0  # penalty
        
        old_side = side
        delta = new_side - old_side
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            side = new_side
            if inner_ok and side < best_side:
                best_side = side
                best_positions = [p[:] for p in positions]
                best_angles = angles[:]
        else:
            # Revert
            positions[idx] = [old_px, old_py]
            angles[idx] = old_angle
        
        T *= cooling
    
    # Also try different outer rotations
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > DEADLINE - 5:
            break
        side = compute_side_length(best_positions, best_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle
    
    # Build result
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
