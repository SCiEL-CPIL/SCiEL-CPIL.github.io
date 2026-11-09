import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3.0)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon with given center, side length, and rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
        # Translate to origin
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        angle_rad = -math.radians(angle_deg)
        rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        
        # Check against 6 half-planes of unit hexagon scaled by side
        # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300
        # The edge normals are at 30, 90, 150, 210, 270, 330
        for k in range(6):
            normal_angle = math.pi / 6 + k * math.pi / 3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            # The support function for a hexagon of side s in direction n is s * sqrt(3) / 2
            # Actually for a regular hexagon with side length s, the inradius is s * sqrt(3)/2
            # The distance from center to each edge is s * sqrt(3)/2
            dot = rx * nx + ry * ny
            if dot > side * sqrt3 / 2.0 + 1e-10:
                return False
        return True
    
    def check_disjoint(hex1, hex2):
        """Check if two hexagons are disjoint (no overlap).
        Each hex is (cx, cy, angle_deg) with side length 1.
        Two unit hexagons are disjoint if the distance between centers is > 2 * inradius
        when aligned, but with rotation it's more complex.
        
        Actually, two regular hexagons of side 1 are disjoint iff no point is in both.
        A sufficient condition: distance between centers > 2 * circumradius = 2.
        But they can be disjoint even closer if rotated.
        
        For a rigorous check: check if any vertex of hex1 is in hex2, any vertex of hex2 is in hex1,
        and no edges intersect.
        
        For simplicity and correctness, let's use a conservative check:
        Two unit hexagons are guaranteed disjoint if center distance > 2.0.
        They might be disjoint even if closer, but let's be safe.
        
        Actually, the inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866.
        Two hexagons with the same orientation are disjoint if center distance > 2 * inradius = sqrt(3) ≈ 1.732.
        With different orientations, the minimum distance for disjointness can be smaller.
        
        Let's do a proper check: two convex polygons are disjoint iff no edge intersects and neither contains a vertex of the other.
        """
        cx1, cy1, a1 = hex1
        cx2, cy2, a2 = hex2
        
        v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
        v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
        
        # Check if any vertex of v1 is in v2
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, cx2, cy2, 1.0, a2):
                return False
        
        # Check if any vertex of v2 is in v1
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, cx1, cy1, 1.0, a1):
                return False
        
        # Check edge intersections
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j + 1) % 6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        
        return True
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        
        # Check collinear cases (touching)
        if d1 == 0 and on_segment(p3, p4, p1):
            return True
        if d2 == 0 and on_segment(p3, p4, p2):
            return True
        if d3 == 0 and on_segment(p1, p2, p3):
            return True
        if d4 == 0 and on_segment(p1, p2, p4):
            return True
        
        return False
    
    def on_segment(a, b, c):
        """Check if point c is on segment a-b (assuming collinear)."""
        return min(a[0], b[0]) - 1e-10 <= c[0] <= max(a[0], b[0]) + 1e-10 and \
               min(a[1], b[1]) - 1e-10 <= c[1] <= max(a[1], b[1]) + 1e-10
    
    def compute_required_side(inner, outer_center, outer_angle):
        """Compute the minimum side length of outer hexagon (at given center and angle) 
        that contains all inner hexagons."""
        ox, oy = outer_center
        max_s = 0.0
        
        for cx, cy, a_deg in inner:
            verts = hexagon_vertices(cx, cy, 1.0, a_deg)
            for vx, vy in verts:
                # Distance from outer center to this vertex
                dx = vx - ox
                dy = vy - oy
                # Rotate into outer hexagon's frame
                angle_rad = -math.radians(outer_angle)
                rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
                ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
                
                # The support function: for each of the 6 edge normals, compute the projection
                for k in range(6):
                    normal_angle = math.pi / 6 + k * math.pi / 3.0
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    dot = rx * nx + ry * ny
                    if dot > 0:
                        # Required side length such that dot <= s * sqrt(3)/2
                        s_req = dot * 2.0 / sqrt3
                        if s_req > max_s:
                            max_s = s_req
        
        return max_s
    
    # Strategy: Use a hexagonal cluster arrangement but optimize positions and rotations
    # via simulated annealing
    
    # Initial arrangement: 12 hexagons in a compact cluster
    # Use axial coordinates for a hex grid
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    # 12 hexagons: center + 6 neighbors + 5 more
    axial_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (2, -1), (1, 1),
        (-1, 2), (-2, 1),
    ]
    
    # Initial inner hexagons
    inner = []
    for q, r in axial_positions:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    # Initial outer parameters
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Check initial validity
    def is_valid(inner, outer_center, outer_angle):
        # Check all inner hexagons are pairwise disjoint
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if not check_disjoint(inner[i], inner[j]):
                    return False
        # Check all inner hexagons are inside outer
        side = compute_required_side(inner, outer_center, outer_angle)
        return True
    
    # Simulated annealing to optimize
    best_score = compute_required_side(inner, outer_center, outer_angle)
    best_inner = [list(h) for h in inner]
    best_center = list(outer_center)
    best_angle = outer_angle
    
    # Check if initial is valid
    if not is_valid(inner, outer_center, outer_angle):
        # Try to fix by scaling
        scale = 1.05
        for h in inner:
            h[0] *= scale
            h[1] *= scale
        best_score = compute_required_side(inner, outer_center, outer_angle)
        best_inner = [list(h) for h in inner]
        best_center = list(outer_center)
        best_angle = outer_angle
    
    # SA parameters
    T = 0.1
    cooling = 0.999
    max_iters = 20000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate a perturbation
        if it % 10 == 0:
            # Move a random inner hexagon
            idx = it % len(inner)
            inner[idx][0] += (random.random() - 0.5) * 0.3
            inner[idx][1] += (random.random() - 0.5) * 0.3
        elif it % 10 == 1:
            # Rotate a random inner hexagon
            idx = it % len(inner)
            inner[idx][2] += (random.random() - 0.5) * 10.0
        elif it % 10 == 2:
            # Move outer center
            outer_center[0] += (random.random() - 0.5) * 0.1
            outer_center[1] += (random.random() - 0.5) * 0.1
        elif it % 10 == 3:
            # Rotate outer
            outer_angle += (random.random() - 0.5) * 5.0
        else:
            # Small random move
            idx = it % len(inner)
            inner[idx][0] += (random.random() - 0.5) * 0.1
            inner[idx][1] += (random.random() - 0.5) * 0.1
            inner[idx][2] += (random.random() - 0.5) * 5.0
        
        # Check validity
        valid = True
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if not check_disjoint(inner[i], inner[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            new_score = compute_required_side(inner, outer_center, outer_angle)
            if new_score < best_score:
                best_score = new_score
                best_inner = [list(h) for h in inner]
                best_center = list(outer_center)
                best_angle = outer_angle
        else:
            # Revert
            # We need to track the previous state
            pass
        
        T *= cooling
    
    # Use best found
    inner = best_inner
    outer_center = best_center
    outer_angle = best_angle
    
    # Add small margin for safety
    side = best_score * 1.005
    
    return inner, (outer_center[0], outer_center[1]), side, outer_angle

import random
# EVOLVE-BLOCK-END
