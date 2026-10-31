import numpy as np
import time
import random
from math import sqrt, cos, sin, pi

N = 11

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon with given center, rotation, and side length."""
    angle_rad = angle_deg * pi / 180.0
    # For a regular hexagon with side length s, the circumradius is s
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees (plus rotation)
    verts = []
    for i in range(6):
        theta = angle_rad + i * pi / 3
        x = cx + side * cos(theta)
        y = cy + side * sin(theta)
        verts.append((x, y))
    return verts

def hexagon_inradius(cx, cy, angle_deg, side=1.0):
    """Inradius (apothem) of a regular hexagon."""
    return side * sqrt(3) / 2

def point_in_hexagon(px, py, hex_center, hex_angle, hex_side=1.0):
    """Check if point is inside a regular hexagon."""
    cx, cy = hex_center
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = -hex_angle * pi / 180.0
    lx = dx * cos(angle_rad) - dy * sin(angle_rad)
    ly = dx * sin(angle_rad) + dy * cos(angle_rad)
    
    # For a regular hexagon with circumradius r=1, the boundaries are at
    # |x| <= 1, |x*cos(60) + y*sin(60)| <= 1, |-x*cos(60) + y*sin(60)| <= 1
    # Actually, for a hexagon with vertices at (1,0), (0.5, sqrt(3)/2), etc.
    # The half-width in x direction is 1, half-height in y is sqrt(3)/2
    # But the sides are at 60 degree angles
    
    # A regular hexagon with circumradius R=1 has:
    # |x| <= 1
    # |x*cos(60) + y*sin(60)| <= 1  => |x/2 + y*sqrt(3)/2| <= 1
    # |-x*cos(60) + y*sin(60)| <= 1 => |-x/2 + y*sqrt(3)/2| <= 1
    # Which simplifies to: |x| <= 1, |y| <= sqrt(3) (no that's wrong)
    
    # Let me think again. A regular hexagon centered at origin with one vertex at (1,0):
    # Vertices: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # The sides are:
    # From (1,0) to (1/2, sqrt(3)/2): line x + sqrt(3)*y = 1... no
    # Actually the constraint for a regular hexagon with circumradius R is:
    # max(|x|, |x*cos(60)+y*sin(60)|, |-x*cos(60)+y*sin(60)|) <= R
    # Wait, that's for a hexagon aligned with axes differently.
    
    # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (circumradius R=1):
    # The supporting lines are at angles 30, 90, 150, 210, 270, 330 with distance sqrt(3)/2 from center
    # So the constraint is: for each of 6 directions, dot product <= sqrt(3)/2
    # Directions: (cos(30),sin(30)), (cos(90),sin(90)), etc.
    
    # Simpler: the hexagon is the intersection of 6 half-planes.
    # For a hexagon with circumradius R, inradius r = R*sqrt(3)/2
    # The 6 normal directions are at angles 30, 90, 150, 210, 270, 330 (if vertex at 0)
    # Actually if vertex at angle 0, normals are at 30, 90, 150, 210, 270, 330
    
    r = hex_side * sqrt(3) / 2
    for i in range(6):
        normal_angle = (i * 60 + 30) * pi / 180  # normals at 30, 90, 150, ...
        nx = cos(normal_angle)
        ny = sin(normal_angle)
        if lx * nx + ly * ny > r + 1e-9:
            return False
    return True

def enclosing_hexagon_size(positions, angles, outer_center, outer_angle):
    """
    Compute the minimum side length of a regular hexagon with given center and angle
    that encloses all unit hexagons.
    """
    ocx, ocy = outer_center
    oa = outer_angle * pi / 180.0
    
    max_required = 0.0
    for (cx, cy), ang in zip(positions, angles):
        # Get vertices of this inner hexagon
        verts = get_hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            # Transform to outer hexagon's local frame
            dx = vx - ocx
            dy = vy - ocy
            lx = dx * cos(-oa) - dy * sin(-oa)
            ly = dx * sin(-oa) + dy * cos(-oa)
            
            # For a regular hexagon with circumradius R, a point (lx,ly) is inside
            # if max over 6 directions of dot product <= R * cos(30) = R * sqrt(3)/2
            # The 6 normal directions for a hexagon with vertex at angle 0 are at 30, 90, 150, 210, 270, 330
            for i in range(6):
                normal_angle = (i * 60 + 30) * pi / 180
                nx = cos(normal_angle)
                ny = sin(normal_angle)
                proj = lx * nx + ly * ny
                if proj > max_required:
                    max_required = proj
    
    # max_required is the max projection onto the 6 normal directions
    # This equals R * sqrt(3)/2 where R is circumradius = side length
    # So side length = max_required / (sqrt(3)/2) = 2 * max_required / sqrt(3)
    side = 2 * max_required / sqrt(3)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    
    # Initial arrangement: hexagonal lattice pattern
    # 1 center + 6 in first ring + 4 in second ring
    positions = []
    angles = []
    
    # Center
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # First ring: 6 hexagons at distance sqrt(3) from center
    d1 = sqrt(3)
    for i in range(6):
        theta = i * pi / 3
        x = d1 * cos(theta)
        y = d1 * sin(theta)
        positions.append((x, y))
        angles.append(0.0)
    
    # Second ring: 4 hexagons at distance 2*sqrt(3), choose 4 of 6 positions
    d2 = 2 * sqrt(3)
    # Place them at angles 0, 60, 120, 180 (skip 240, 300 for compactness)
    for i in [0, 1, 2, 3]:
        theta = i * pi / 3
        x = d2 * cos(theta)
        y = d2 * sin(theta)
        positions.append((x, y))
        angles.append(0.0)
    
    # Verify no overlaps initially
    def check_overlaps(pos, ang):
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = sqrt(dx*dx + dy*dy)
                # Two unit hexagons (same orientation) overlap if distance < sqrt(3)
                # With different orientations, minimum distance is still at least sqrt(3) for no overlap
                # Actually the minimum distance between centers for non-overlapping unit hexagons
                # depends on relative rotation. The worst case (most overlap) is when they're aligned,
                # requiring distance >= sqrt(3). With rotation, it can be less.
                # Conservative: require distance >= sqrt(3) - epsilon
                if dist < sqrt(3) - 0.01:
                    return False
        return True
    
    # Optimize using simple local search
    best_side = enclosing_hexagon_size(positions, angles, (0, 0), 0.0)
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    
    step_size = 0.5
    temp = 2.0
    
    while time.time() < deadline:
        # Pick a random hexagon to move
        idx = random.randint(0, N-1)
        old_x, old_y = positions[idx]
        old_ang = angles[idx]
        
        # Random perturbation
        positions[idx] = (old_x + random.gauss(0, step_size), 
                         old_y + random.gauss(0, step_size))
        angles[idx] = old_ang + random.gauss(0, 10)
        
        # Check overlaps
        if not check_overlaps(positions, angles):
            positions[idx] = (old_x, old_y)
            angles[idx] = old_ang
            temp *= 0.95
            continue
        
        new_side = enclosing_hexagon_size(positions, angles, (0, 0), 0.0)
        
        if new_side < best_side:
            best_side = new_side
            best_positions = [p[:] for p in positions]
            best_angles = angles[:]
        else:
            # Revert
            positions[idx] = (old_x, old_y)
            angles[idx] = old_ang
        
        temp *= 0.99
        if temp < 0.1:
            break
    
    # Use best found
    positions = best_positions
    angles = best_angles
    
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, (0.0, 0.0), best_side, 0.0

