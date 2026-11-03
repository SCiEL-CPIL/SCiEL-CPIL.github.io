import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Define the 12 inner hexagon centers in a 3x4 honeycomb grid
    # Hexagon side length is 1.
    # Horizontal distance between centers in same row: sqrt(3)
    # Vertical distance between rows: 1.5
    # Horizontal offset between rows: sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    
    # Generate raw centers
    centers = []
    # Row 0: 4 hexagons
    for j in range(4):
        centers.append((j * sqrt3, 0.0))
    # Row 1: 4 hexagons
    for j in range(4):
        centers.append(((j + 0.5) * sqrt3, 1.5))
    # Row 2: 4 hexagons
    for j in range(4):
        centers.append((j * sqrt3, 3.0))
    
    # Calculate centroid to center the cluster at (0,0)
    cx = sum(p[0] for p in centers) / len(centers)
    cy = sum(p[1] for p in centers) / len(centers)
    
    # Center the cluster
    centers = [(p[0] - cx, p[1] - cy) for p in centers]
    
    # 2. Define vertices of a unit hexagon centered at origin, angle 0
    # Vertices are at radius 1, angles 0, 60, ..., 300
    # Wait, standard hexagon with flat top/bottom or pointy top?
    # If side is 1, radius is 1.
    # If angle 0 means a vertex is on the x-axis (pointy right), vertices are at 0, 60, ...
    # If angle 0 means a side is on the x-axis (flat right), vertices are at 30, 90, ...
    # The problem says "angle_degrees". Usually 0 is standard orientation.
    # Let's assume 0 degrees means a vertex is at (1, 0).
    # This is "pointy" orientation.
    # However, honeycomb packing usually aligns flat sides.
    # If we rotate inner hexagons by 30 degrees, they are "flat" oriented.
    # Let's try both or just pick one. 
    # Actually, for a regular hexagon, the "shape" is invariant under 60 deg rotation.
    # The bounding box depends on the orientation relative to the outer hexagon.
    # We can fix inner angle to 0 (vertex at 1,0) and let the outer hexagon rotate.
    # Or fix inner angle to 30 (flat side) and let outer rotate.
    # It doesn't matter for the relative rotation.
    
    # Let's use vertices for a hexagon with side 1, centered at (0,0), angle 0 (vertex at 1,0).
    # v_k = (cos(k*60), sin(k*60)) for k=0..5
    base_vertices = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))) for k in range(6)]
    
    # Pre-compute vertices for all 12 inner hexagons
    # Since inner hexagons are all aligned (angle 0), we just translate the base vertices
    inner_hex_vertices = []
    for cx_i, cy_i in centers:
        verts = [(vx + cx_i, vy + cy_i) for vx, vy in base_vertices]
        inner_hex_vertices.append(verts)
        
    # 3. Optimize outer hexagon rotation and side length
    # Outer hexagon centered at (0,0).
    # Side length S. Apothem a = S * sqrt(3) / 2.
    # Normals of outer hexagon sides:
    # If outer hexagon has rotation theta, its normals are at theta, theta+60, ..., theta+300.
    # Condition for point (x,y) to be inside outer hexagon:
    # For all 6 normals n_k, x*n_kx + y*n_ky <= a.
    # So a >= max( x*n_kx + y*n_ky ) for all points and all k.
    # S = 2 * a / sqrt(3).
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Scan theta from 0 to 60 degrees
    # Step size 0.5 degrees is likely sufficient for high precision
    num_steps = 120
    for i in range(num_steps + 1):
        theta_deg = (i * 60.0) / num_steps
        theta_rad = math.radians(theta_deg)
        
        # Calculate max projection on all 6 normals
        max_proj = 0.0
        for k in range(6):
            phi = theta_rad + k * math.pi / 3
            nx = math.cos(phi)
            ny = math.sin(phi)
            
            # Find max projection of any vertex of any inner hexagon on this normal
            # We can iterate through all vertices
            for verts in inner_hex_vertices:
                for vx, vy in verts:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
                        
        # max_proj is the required apothem 'a'
        a = max_proj
        S = 2.0 * a / sqrt3
        
        if S < best_S:
            best_S = S
            best_theta = theta_deg
            
        if time.time() > deadline:
            break

    # Final refinement around best_theta with smaller steps
    if time.time() < deadline - 5.0:
        ref_steps = 200
        range_deg = 1.0 # +/- 1 degree
        for i in range(ref_steps + 1):
            theta_deg = best_theta - range_deg + (i * 2.0 * range_deg) / ref_steps
            theta_rad = math.radians(theta_deg)
            
            max_proj = 0.0
            for k in range(6):
                phi = theta_rad + k * math.pi / 3
                nx = math.cos(phi)
                ny = math.sin(phi)
                
                for verts in inner_hex_vertices:
                    for vx, vy in verts:
                        proj = vx * nx + vy * ny
                        if proj > max_proj:
                            max_proj = proj
                            
            a = max_proj
            S = 2.0 * a / sqrt3
            
            if S < best_S:
                best_S = S
                best_theta = theta_deg

    # Construct return values
    # inner: list of [x, y, angle_degrees]
    inner = [[cx_i, cy_i, 0.0] for cx_i, cy_i in centers]
    center = [0.0, 0.0]
    side = best_S
    angle = best_theta

    return inner, center, side, angle
# EVOLVE-BLOCK-END
