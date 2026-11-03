"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s out of 120s budget
    
    # Unit hexagon vertices (flat-top)
    # Center at origin
    vertices = [(1.0, 0.0), (0.5, math.sqrt(3)/2), (-0.5, math.sqrt(3)/2),
                (-1.0, 0.0), (-0.5, -math.sqrt(3)/2), (0.5, -math.sqrt(3)/2)]
    
    def rotate_point(x, y, angle_rad):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return (x*c - y*s, x*s + y*c)
    
    def hexagon_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        vs = []
        for vx, vy in vertices:
            rx, ry = rotate_point(vx, vy, angle_rad)
            vs.append((cx + rx, cy + ry))
        return vs
    
    def min_outer_hex_side(inner_points, outer_center, outer_angle_deg):
        """
        Find the minimum side length S of a regular hexagon with center outer_center
        and rotation outer_angle_deg that contains all inner_points.
        
        A regular hexagon with side S and angle theta contains point p iff
        for each of the 6 half-planes defined by the sides, p is on the inside.
        
        The hexagon can be described by 6 supporting lines. The distance from center
        to each side (apothem) is S * sqrt(3) / 2.
        """
        theta = math.radians(outer_angle_deg)
        max_apothem = 0.0
        for i in range(6):
            # Normal direction for side i
            # Side i is perpendicular to direction (theta + i*pi/3)
            # Actually, for a flat-top hexagon (angle=0), sides are at angles 0, 60, 120, 180, 240, 300
            # The apothem direction for side i is at angle theta + i*pi/3
            # Wait, let's think carefully.
            # A regular hexagon with "flat top" (vertex at 90 deg? No, flat top means a side is horizontal at top).
            # Let's use the standard: vertices at angles 0, 60, 120, 180, 240, 300.
            # Sides are between consecutive vertices. The normal to side i (between vertex i and i+1)
            # is at angle (0 + 60*(i+0.5)) = 30 + 60*i degrees.
            # The apothem (center to side midpoint) is in the direction of this normal.
            # For a point to be inside, its projection on this normal must be <= apothem.
            normal_angle = theta + math.pi/6 + i * math.pi/3
            nx, ny = math.cos(normal_angle), math.sin(normal_angle)
            max_proj = 0.0
            for px, py in inner_points:
                # Vector from outer center to point
                dx = px - outer_center[0]
                dy = py - outer_center[1]
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
            # apothem = S * sqrt(3) / 2
            # We need apothem >= max_proj
            # S >= max_proj * 2 / sqrt(3)
            required_S = max_proj * 2.0 / math.sqrt(3)
            if required_S > max_apothem:
                max_apothem = required_S
        return max_apothem
    
    def check_validity(inner_hexes, outer_center, outer_side, outer_angle):
        """Check that all inner hexagons are inside the outer hexagon and don't overlap."""
        # Check overlap first (simpler)
        for i in range(len(inner_hexes)):
            for j in range(i+1, len(inner_hexes)):
                if hexagons_overlap(inner_hexes[i], inner_hexes[j]):
                    return False
        # Check containment
        for hx, hy, ha in inner_hexes:
            vs = hexagon_vertices(hx, hy, ha)
            for vx, vy in vs:
                if not point_in_hexagon(vx, vy, outer_center, outer_side, outer_angle):
                    return False
        return True
    
    def hexagons_overlap(h1, h2):
        """Check if two hexagons overlap using vertex inclusion or edge intersection."""
        # Simplified: check if any vertex of one is inside the other
        vs1 = hexagon_vertices(h1[0], h1[1], h1[2])
        vs2 = hexagon_vertices(h2[0], h2[1], h2[2])
        for v in vs1:
            if point_in_convex_polygon(v[0], v[1], vs2):
                return True
        for v in vs2:
            if point_in_convex_polygon(v[0], v[1], vs1):
                return True
        return False
    
    def point_in_convex_polygon(px, py, polygon):
        """Check if point is inside convex polygon (CCW or CW)."""
        n = len(polygon)
        sign = 0
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i+1) % n]
            cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
            if cross > 1e-10:
                if sign < 0: return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0: return False
                sign = -1
        return True
    
    def point_in_hexagon(px, py, center, side, angle_deg):
        """Check if point is inside regular hexagon."""
        theta = math.radians(angle_deg)
        # The hexagon is the intersection of 6 half-planes
        for i in range(6):
            normal_angle = theta + math.pi/6 + i * math.pi/3
            nx, ny = math.cos(normal_angle), math.sin(normal_angle)
            dx = px - center[0]
            dy = py - center[1]
            proj = dx * nx + dy * ny
            apothem = side * math.sqrt(3) / 2
            if proj > apothem + 1e-10:
                return False
        return True
    
    # Generate initial configurations
    
    # Configuration 1: Linear arrangement (from seed)
    def config_linear():
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Configuration 2: Honeycomb cluster
    def config_honeycomb():
        # Hexagonal packing
        sqrt3 = math.sqrt(3)
        inner = []
        # Center
        inner.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons
        for i in range(6):
            angle = math.radians(60 * i)
            x = sqrt3 * math.cos(angle)
            y = sqrt3 * math.sin(angle)
            inner.append([x, y, 0.0])
        # Ring 2: 4 more to make 11 total (need 4 more)
        # Add at positions that continue the pattern
        # Positions at distance 2*sqrt3 from center
        for i in range(6):
            angle = math.radians(60 * i)
            x = 2 * sqrt3 * math.cos(angle)
            y = 2 * sqrt3 * math.sin(angle)
            inner.append([x, y, 0.0])
        # Take first 11
        inner = inner[:11]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    # Configuration 3: Rectangular grid
    def config_grid():
        sqrt3 = math.sqrt(3)
        inner = []
        # 4x3 grid = 12, take 11
        for row in range(3):
            for col in range(4):
                x = col * 2.0  # spacing between centers in x direction
                y = row * sqrt3  # spacing in y direction
                # Offset alternate rows
                if row % 2 == 1:
                    x += 1.0
                inner.append([x, y, 0.0])
        inner = inner[:11]
        # Center the cluster
        avg_x = sum(p[0] for p in inner) / len(inner)
        avg_y = sum(p[1] for p in inner) / len(inner)
        inner = [[p[0] - avg_x, p[1] - avg_y, p[2]] for p in inner]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    # Configuration 4: Star pattern
    def config_star():
        sqrt3 = math.sqrt(3)
        inner = [[0.0, 0.0, 0.0]]
        # 6 around center
        for i in range(6):
            angle = math.radians(60 * i)
            x = sqrt3 * math.cos(angle)
            y = sqrt3 * math.sin(angle)
            inner.append([x, y, 0.0])
        # 4 more at cardinal directions further out
        for i in range(4):
            angle = math.radians(90 * i + 45)
            x = 2.5 * math.cos(angle)
            y = 2.5 * math.sin(angle)
            inner.append([x, y, 0.0])
        inner = inner[:11]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    configs = [config_linear, config_honeycomb, config_grid, config_star]
    
    best_score = float('inf')
    best_result = None
    
    for cfg_func in configs:
        if time.time() > deadline:
            break
        inner, center, side, angle = cfg_func()
        
        # Verify validity
        if not check_validity(inner, center, side, angle):
            continue
        
        # Optimize outer hexagon parameters
        # Try different outer angles and centers
        best_for_config = side
        
        for outer_angle in [0, 15, 30, 45, 60, 75, 90]:
            if time.time() > deadline:
                break
            
            # Try to find minimal side by optimizing center
            # For simplicity, use center of mass
            avg_x = sum(p[0] for p in inner) / len(inner)
            avg_y = sum(p[1] for p in inner) / len(inner)
            
            for dx in [-1, -0.5, 0, 0.5, 1]:
                for dy in [-1, -0.5, 0, 0.5, 1]:
                    c = (avg_x + dx, avg_y + dy)
                    s = min_outer_hex_side(
                        [(p[0], p[1]) for p in inner],
                        c,
                        outer_angle
                    )
                    s += 1e-9  # small epsilon for safety
                    
                    if s < best_for_config:
                        # Check validity with this outer hexagon
                        if check_validity(inner, c, s, outer_angle):
                            best_for_config = s
        
        if best_for_config < best_score:
            best_score = best_for_config
            best_result = (inner, (0.0, 0.0), best_for_config, 0.0)
    
    # If no valid configuration found, return the linear one with large side
    if best_result is None:
        inner, center, side, angle = config_linear()
        # Make side large enough to definitely contain
        side = 50.0
        best_result = (inner, center, side, angle)
    
    return best_result
# EVOLVE-BLOCK-END
