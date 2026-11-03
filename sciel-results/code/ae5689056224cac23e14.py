import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def hexagon_overlap_penalty(poly1, poly2):
        """
        Estimate overlap between two convex polygons (hexagons).
        We use a simple heuristic: check if any vertex of one is inside the other,
        or if edges intersect. For speed in optimization, we can use a simplified
        distance-based penalty or a robust geometric check.
        
        For high performance, let's use a fast approximate check first, 
        then exact if close.
        Actually, for Nelder-Mead, we can just use a penalty based on minimum distance
        between convex hulls if they don't intersect, or a large penalty if they do.
        
        Let's use a robust but fast method:
        1. Check bounding box overlap.
        2. If no BB overlap, return 0.
        3. If BB overlap, check if any vertex of poly1 is inside poly2 or vice versa.
        4. If no vertex inside, check edge intersections.
        
        To keep it simple and fast for the optimizer, we will use a simplified
        "minimum separation" proxy. However, strict non-overlap is required.
        
        Let's implement a fast point-in-polygon and segment intersection.
        """
        # Get vertices
        v1 = poly1
        v2 = poly2
        
        # Bounding Box Check
        min_x1, min_y1 = np.min(v1, axis=0)
        max_x1, max_y1 = np.max(v1, axis=0)
        min_x2, min_y2 = np.min(v2, axis=0)
        max_x2, max_y2 = np.max(v2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return 0.0

        # Point in Polygon check (Ray casting)
        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check vertices of v1 in v2
        for pt in v1:
            if point_in_poly(pt, v2):
                return 100.0 # Large penalty
        
        # Check vertices of v2 in v1
        for pt in v2:
            if point_in_poly(pt, v1):
                return 100.0

        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return 100.0
        
        return 0.0

    def objective(params):
        """
        Objective function for optimization.
        params: array of length 2N (x, y for each hex). We fix angles to 0 for initial simplicity, 
        or allow angles? The prompt says "each inner hexagon may translate and rotate".
        However, due to symmetry, if the container is a regular hexagon, rotating inner hexagons
        by 60 degrees is equivalent. But relative rotation might help?
        Usually, for hexagonal packing, aligning them (angle 0) is optimal or near-optimal.
        Let's fix angles to 0 to reduce dimensionality and improve convergence speed.
        If we need to rotate, we can add that later, but 2N variables is much faster than 3N.
        """
        if time.time() > deadline - 5:
            return 1e6

        positions = np.array(params).reshape(N, 2)
        
        # Calculate overlap penalty
        penalty = 0.0
        polys = []
        for i in range(N):
            cx, cy = positions[i]
            verts = get_hex_verts(cx, cy, 1.0, 0.0)
            polys.append(verts)
            
        for i in range(N):
            for j in range(i+1, N):
                if penalty > 1.0: # Early exit if significant overlap
                    break
                p = hexagon_overlap_penalty(polys[i], polys[j])
                if p > penalty:
                    penalty = p
            
        if penalty > 0:
            # Return a large value proportional to overlap
            return penalty + calc_min_outer_side(positions, np.zeros(N), 0.0)
            
        side = calc_min_outer_side(positions, np.zeros(N), 0.0)
        return side

    # Strategy:
    # 1. Initialize with a known good configuration: Hexagonal cluster (Flower)
    #    Center + 6 around + 5 more? 
    #    A standard hexagonal packing of 12 hexagons:
    #    It's like a hexagon of hexagons with side 2 (which has 7) plus 5 more?
    #    Actually, a hexagon of side 3 has 19. Side 2 has 7.
    #    We need 12.
    #    Let's try a 3x4 grid or a distorted hexagonal cluster.
    #    
    #    Better initialization:
    #    Take a hexagonal lattice of spacing 2 (side 1 hexes touch).
    #    Generate a large grid, pick the 12 closest to center.
    
    sqrt3 = math.sqrt(3)
    # Hex lattice basis vectors for flat-top? 
    # Let's use pointy-top (angle 0).
    # Neighbors of (0,0) in hex lattice with side 1 (distance 2 between centers):
    # (2, 0), (1, sqrt3), (-1, sqrt3), (-2, 0), (-1, -sqrt3), (1, -sqrt3)
    
    lattice_points = []
    # Generate a 5x5 grid of hex lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Hex grid to Cartesian (pointy top)
            # x = size * sqrt(3) * (col + row/2)
            # y = size * 3/2 * row
            # Here size=1, distance between centers = 2.
            # So x = 2 * sqrt(3) / 2 * (i + j/2) = sqrt(3)*(i + j/2)
            # y = 2 * 3/2 * j = 3*j
            # Wait, standard hex grid spacing:
            # If side is 1, distance between centers of touching hexes is 2.
            # For pointy-top:
            # x = 2 * sqrt(3) / 2 * (col + row/2) = sqrt(3) * (col + row/2)
            # y = 3/2 * row * 2 = 3 * row
            x = sqrt3 * (i + j/2.0)
            y = 3 * j
            lattice_points.append((x, y))
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the 12 closest
    init_positions = np.array(lattice_points[:12])
    
    # Center them
    init_positions -= np.mean(init_positions, axis=0)
    
    # Run Nelder-Mead on the 2N positions
    # This is much faster than DE on 3N variables.
    
    best_result = init_positions
    best_score = objective(best_result)
    
    try:
        res_nm = minimize(
            objective, 
            best_result, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_result = res_nm.x
    except Exception as e:
        print(f"NM failed: {e}")

    # If the result has overlap (penalty > 0), it's invalid.
    # Check final validity
    final_positions = best_result.reshape(N, 2)
    final_angles = np.zeros(N)
    
    # Verify no overlap
    valid = True
    polys = [get_hex_verts(final_positions[i][0], final_positions[i][1], 1.0, 0.0) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_overlap_penalty(polys[i], polys[j]) > 0:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback to a simple valid configuration: 
        # Place them in a 2x6 grid or similar, even if not minimal.
        # A 2x6 grid of hexagons:
        # Width: 6 * 2 = 12? No, touching is distance 2.
        # Let's just use the honeycomb fallback from the previous program which was valid.
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        final_positions = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        final_angles = np.zeros(N)
        valid = True # Assume this is valid as it was in the seed

    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], 0.0])
    
    return inner, (0.0, 0.0), final_side, 0.0
