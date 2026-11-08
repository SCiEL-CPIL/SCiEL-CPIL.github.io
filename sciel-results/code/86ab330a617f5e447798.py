import math
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Configuration: 3 rows, 4 columns of Pointy-Top hexagons
    # Inner Hexagons are Pointy-Top (Angle 0)
    # Outer Hexagon is Pointy-Top (Angle 0)
    
    # Grid parameters for Pointy-Top hexagons (Side 1)
    # Width = sqrt(3), Height = 2
    # Horizontal spacing = sqrt(3)
    # Vertical spacing = 1.5
    # Horizontal offset for alternating rows = sqrt(3)/2
    
    # Add a small scale factor to ensure strict disjointness (gap)
    # This prevents floating point issues or strict inequality checks in the evaluator
    scale = 1.0001
    
    h_spacing = math.sqrt(3) * scale
    v_spacing = 1.5 * scale
    h_offset = (math.sqrt(3) / 2.0) * scale
    
    inner = []
    
    # Generate 12 centers: 3 rows (r=0,1,2), 4 cols (c=0,1,2,3)
    # Row 0 (r=0): offset 0
    # Row 1 (r=1): offset h_offset
    # Row 2 (r=2): offset 0
    
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * h_spacing
            y = r * v_spacing
            if r % 2 != 0:
                x += h_offset
            centers.append((x, y))
            
    # Calculate bounding box of centers to center the lattice at (0,0)
    min_x = min(p[0] for p in centers)
    max_x = max(p[0] for p in centers)
    min_y = min(p[1] for p in centers)
    max_y = max(p[1] for p in centers)
    
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    for x, y in centers:
        inner.append([x - cx, y - cy, 0.0])
        
    # Outer Hexagon
    # We need to determine the minimum side length S such that all inner hexagons fit.
    # Inner Hexagons are Pointy-Top.
    # Outer Hexagon is Pointy-Top.
    
    # For a Pointy-Top hexagon of side S centered at origin:
    # The normals to the sides are at 0, 60, 120, 180, 240, 300 degrees.
    # The apothem (distance from center to side) is S * sqrt(3) / 2.
    
    # We check the maximum projection of all vertices of all inner hexagons
    # onto these 6 normal directions.
    
    # Try a few rotations for the outer hexagon to find the minimal S
    best_S = float('inf')
    best_angle = 0.0
    
    # The lattice is symmetric, so 0 to 30 degrees is sufficient, but let's check 0 to 60
    for out_angle_deg in range(0, 61, 5):
        out_angle_rad = math.radians(out_angle_deg)
        
        normals = []
        for k in range(6):
            angle_rad = out_angle_rad + math.radians(60 * k)
            normals.append((math.cos(angle_rad), math.sin(angle_rad)))
        
        max_proj = 0.0
        for cx_hex, cy_hex, _ in inner:
            # Generate vertices of this inner hexagon (side 1, Pointy-Top)
            # Vertices are at 30, 90, 150, 210, 270, 330 degrees
            for v_angle_deg in range(30, 360, 60):
                v_angle_rad = math.radians(v_angle_deg)
                vx = cx_hex + math.cos(v_angle_rad)
                vy = cy_hex + math.sin(v_angle_rad)
                for nx, ny in normals:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
                        
        # S * sqrt(3) / 2 >= max_proj  => S >= max_proj * 2 / sqrt(3)
        S = max_proj * 2.0 / math.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_angle = out_angle_deg
            
    # Add a tiny epsilon to ensure strict inequality if needed
    best_S += 1e-9
    
    center = (0.0, 0.0)
    angle = best_angle
    
    return inner, center, best_S, angle

