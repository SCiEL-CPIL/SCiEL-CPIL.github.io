import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_hex_vertices(center, side, angle_deg):
        cx, cy = center
        ang = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = ang + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return verts

    def polygon_overlap(v1, v2):
        # Simple separating axis theorem check for convex polygons
        def project(pts, axis):
            return [p[0] * axis[0] + p[1] * axis[1] for p in pts]
        
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-9:
                axes.append((normal[0]/length, normal[1]/length))
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-9:
                axes.append((normal[0]/length, normal[1]/length))
                
        for axis in axes:
            proj1 = project(v1, axis)
            proj2 = project(v2, axis)
            max1 = max(proj1)
            min1 = min(proj1)
            max2 = max(proj2)
            min2 = min(proj2)
            if max1 < min2 or max2 < min1:
                return False
        return True

    def packings_overlap(pack):
        # pack is list of (x, y, angle)
        verts_list = []
        for x, y, a in pack:
            verts_list.append(get_hex_vertices((x, y), 1.0, a))
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                if polygon_overlap(verts_list[i], verts_list[j]):
                    return True
        return False

    def min_enclosing_hex_side(pack, outer_angle_deg):
        # Find min side length S and center (cx, cy) for outer hexagon
        # such that all inner hexagons are inside.
        # We can fix outer center to centroid for symmetry, or optimize.
        # For simplicity and speed, we assume outer center is centroid of inner centers
        # and optimize S and outer_angle.
        
        cx = sum(p[0] for p in pack) / len(pack)
        cy = sum(p[1] for p in pack) / len(pack)
        
        def required_side(outer_angle):
            max_dist = 0.0
            for x, y, a in pack:
                verts = get_hex_vertices((x, y), 1.0, a)
                for vx, vy in verts:
                    # Distance from outer center to vertex
                    dx = vx - cx
                    dy = vy - cy
                    # Project onto the 6 normal directions of the outer hexagon
                    # The outer hexagon is defined by 6 lines.
                    # A point is inside if its projection on each normal <= apothem_outer
                    # apothem_outer = S * sqrt(3)/2
                    # So we need max projection <= S * sqrt(3)/2
                    # S >= max_proj / (sqrt(3)/2)
                    for k in range(6):
                        ang = math.radians(outer_angle) + math.pi / 6.0 + k * math.pi / 3.0
                        nx = math.cos(ang)
                        ny = math.sin(ang)
                        proj = dx * nx + dy * ny
                        # We need |proj| <= S * APOTHEM
                        # So S >= |proj| / APOTHEM
                        max_dist = max(max_dist, abs(proj))
            return max_dist / APOTHEM

        # Optimize outer_angle
        best_S = float('inf')
        best_angle = 0.0
        for i in range(0, 60, 1): # 1 degree steps
            ang = float(i)
            S = required_side(ang)
            if S < best_S:
                best_S = S
                best_angle = ang
        
        # Refine
        for i in range(-10, 11):
            ang = best_angle + i * 0.1
            S = required_side(ang)
            if S < best_S:
                best_S = S
                best_angle = ang

        return best_S, (cx, cy), best_angle

    # Generate candidate packings
    # 1. Hexagonal grid pattern
    # Center at (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: Select 4 to make it compact
    
    candidates = []
    
    # Pattern 1: Central + 6 + 4 (from ring 2, symmetric)
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    # Distances in grid units: 1 unit = sqrt(3) distance? No, side 1 hexagons touch at distance sqrt(3).
    # Grid step is sqrt(3) in x and 1.5 in y for axial (1,0) and (0,1).
    # Let's just place them explicitly.
    
    # Center
    pts = [(0.0, 0.0)]
    # Ring 1 (distance sqrt(3))
    for i in range(6):
        ang = i * math.pi / 3
        x = math.sqrt(3) * math.cos(ang)
        y = math.sqrt(3) * math.sin(ang)
        pts.append((x, y))
    # Ring 2 (distance 2*sqrt(3) from center? No, axial (2,0) is 2*sqrt(3) away?
    # Axial (1,1) is distance sqrt(3) * sqrt(1^2 + 1^2/4 + 1*1)?
    # Let's use Cartesian distances for packing.
    # For unit hexagons, min center distance is sqrt(3).
    # Ring 2 centers are at distance 2*sqrt(3) from center if aligned, or sqrt(3)*sqrt(7) etc.
    
    # Let's use a known good set of 11 points that are pairwise >= sqrt(3) apart.
    # The "flower" of 7 is:
    # (0,0)
    # (sqrt(3), 0), (0, sqrt(3)), (-sqrt(3), 0), (0, -sqrt(3)), (sqrt(3)/2, 1.5), (-sqrt(3)/2, -1.5) -- Wait, angles.
    # Standard hex grid neighbors:
    # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1) in axial.
    # Cartesian:
    # (sqrt(3), 0)
    # (sqrt(3)/2, 1.5)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    
    # These 7 are valid.
    # Now add 4 more.
    # Next ring axial: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Cartesian for (2,0): (2*sqrt(3), 0)
    # Cartesian for (1,1): (sqrt(3)*(1+0.5), 1.5) = (1.5*sqrt(3), 1.5)
    # Cartesian for (0,2): (sqrt(3), 3.0)
    # Cartesian for (-1,2): (sqrt(3)*(-1+1), 3.0) = (0, 3.0)
    # Cartesian for (-2,1): (sqrt(3)*(-2+0.5), 1.5) = (-1.5*sqrt(3), 1.5)
    # Cartesian for (-2,0): (-2*sqrt(3), 0)
    
    # Let's pick 4 from the second ring that are far from each other and close to the center to minimize bounding box.
    # Symmetric choice: (2,0), (-2,0), (0,2), (0,-2) in axial?
    # Axial (2,0) -> (2*sqrt(3), 0)
    # Axial (-2,0) -> (-2*sqrt(3), 0)
    # Axial (0,2) -> (sqrt(3), 3) -- Wait, y=1.5*2=3. x=sqrt(3)*(0+1)=sqrt(3).
    # Axial (0,-2) -> (sqrt(3)*(-1), -3) = (-sqrt(3), -3).
    
    # Check distance between (2,0) and (1,1) [which is (1.5*sqrt(3), 1.5)]
    # dx = 0.5*sqrt(3), dy = -1.5. dist^2 = 0.75 + 2.25 = 3. dist = sqrt(3). OK.
    # Check distance between (2,0) and (1,-1) [which is (1.5*sqrt(3), -1.5)]
    # dx = 0.5*sqrt(3), dy = 1.5. dist = sqrt(3). OK.
    
    # So we can add (2,0), (-2,0), (0,2), (0,-2) to the central 7?
    # Let's check if they overlap with each other.
    # (2,0) and (0,2):
    # P1: (2*sqrt(3), 0)
    # P2: (sqrt(3), 3)
    # dx = sqrt(3), dy = -3. dist^2 = 3 + 9 = 12. dist = sqrt(12) > sqrt(3). OK.
    
    # So this set of 11 is valid:
    # 1. (0,0)
    # 2-7. Ring 1
    # 8. (2*sqrt(3), 0)
    # 9. (-2*sqrt(3), 0)
    # 10. (sqrt(3), 3)
    # 11. (-sqrt(3), -3)
    
    pack1 = [
        (0.0, 0.0),
        (math.sqrt(3), 0.0),
        (math.sqrt(3)/2, 1.5),
        (-math.sqrt(3)/2, 1.5),
        (-math.sqrt(3), 0.0),
        (-math.sqrt(3)/2, -1.5),
        (math.sqrt(3)/2, -1.5),
        (2*math.sqrt(3), 0.0),
        (-2*math.sqrt(3), 0.0),
        (math.sqrt(3), 3.0),
        (-math.sqrt(3), -3.0)
    ]
    candidates.append(pack1)
    
    # Pattern 2: A slightly different arrangement, maybe shifting the top/bottom
    # Or just try rotating the whole pack1? The optimizer will handle rotation.
    
    best_S = float('inf')
    best_pack = None
    best_center = None
    best_angle = 0.0
    
    for pack in candidates:
        if packings_overlap(pack):
            continue
        S, center, angle = min_enclosing_hex_side(pack, 0.0)
        if S < best_S:
            best_S = S
            best_pack = pack
            best_center = center
            best_angle = angle
            
    if best_pack is None:
        # Fallback: just one hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inner = [[float(x), float(y), 0.0] for x, y in best_pack]
    cx, cy = best_center
    return inner, [float(cx), float(cy)], float(best_S), float(best_angle)
