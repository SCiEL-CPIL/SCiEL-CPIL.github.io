import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    sqrt3 = math.sqrt(3.0)
    
    inner = []
    
    # Use a more compact arrangement:
    # Place hexagons in a tighter cluster to minimize bounding hexagon
    
    # Strategy: Use 12 hexagons in a compact 3x4 grid-like arrangement
    # with some rotation to fit better in a hexagon
    
    # First, try the standard hexagonal cluster but with optimized spacing
    # Use axial coordinates for flat-top hexagons
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    # Compact arrangement: central + 6 neighbors + 5 from next ring
    # but with slight compression to allow smaller outer hexagon
    positions = [
        (0, 0),      # center
        (1, 0),      # right
        (-1, 0),     # left
        (0, 1),      # upper right
        (0, -1),     # lower right
        (1, -1),     # lower left
        (-1, 1),     # upper left
        (2, 0),      # far right
        (2, -1),     # far lower right
        (1, 1),      # far upper right
        (-1, 2),     # far upper left
        (-2, 1),     # far left
    ]
    
    # Scale positions to ensure strict disjointness (gap between hexagons)
    # Standard distance is sqrt(3). Scaling by 1.01 gives distance 1.01*sqrt(3)
    # which is > sqrt(3), ensuring no overlap.
    scale_factor = 1.01
    
    for q, r in positions:
        x, y = axial_to_cartesian(q, r)
        x *= scale_factor
        y *= scale_factor
        inner.append([x, y, 0.0])
    
    # Now optimize: rotate the outer hexagon to minimize required side length
    # Try different rotation angles for the outer hexagon
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Vertices of a unit flat-top hexagon relative to its center
    hex_verts = [
        (1.0, 0.0),
        (0.5, sqrt3 / 2.0),
        (-0.5, sqrt3 / 2.0),
        (-1.0, 0.0),
        (-0.5, -sqrt3 / 2.0),
        (0.5, -sqrt3 / 2.0)
    ]
    
    # Try 36 different rotations (every 10 degrees)
    for angle_deg in range(0, 60, 10):
        angle_rad = math.radians(angle_deg)
        
        # Normals for the outer hexagon at this rotation
        # For a hexagon with vertex at angle_rad, edge normals are at angle_rad + 30 + k*60
        normals = []
        for k in range(6):
            normal_angle = angle_rad + math.pi / 6 + k * math.pi / 3
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))
        
        max_required_s = 0.0
        
        for cx, cy, _ in inner:
            for vx, vy in hex_verts:
                px = cx + vx
                py = cy + vy
                for nx, ny in normals:
                    val = px * nx + py * ny
                    if val > 0:
                        s_req = val * 2.0 / sqrt3
                        if s_req > max_required_s:
                            max_required_s = s_req
        
        if max_required_s < best_side:
            best_side = max_required_s
            best_angle = angle_deg
    
    # Also try finer granularity around the best angle
    for angle_deg in [best_angle - 5, best_angle - 2, best_angle, best_angle + 2, best_angle + 5]:
        if angle_deg < 0 or angle_deg >= 60:
            continue
        angle_rad = math.radians(angle_deg)
        
        normals = []
        for k in range(6):
            normal_angle = angle_rad + math.pi / 6 + k * math.pi / 3
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))
        
        max_required_s = 0.0
        
        for cx, cy, _ in inner:
            for vx, vy in hex_verts:
                px = cx + vx
                py = cy + vy
                for nx, ny in normals:
                    val = px * nx + py * ny
                    if val > 0:
                        s_req = val * 2.0 / sqrt3
                        if s_req > max_required_s:
                            max_required_s = s_req
        
        if max_required_s < best_side:
            best_side = max_required_s
            best_angle = angle_deg
    
    # Add a small margin for numerical safety
    side = best_side * 1.002
    center = (0.0, 0.0)
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
