import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice
    # Axial coordinates (q, r)
    # Center (0,0)
    # Ring 1 (6 hexagons)
    # Ring 2 partial (4 hexagons)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Ensure we have exactly N
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    elif len(axial_coords) < N:
        # Fallback: add more from ring 2
        # This shouldn't happen for N=11 with the above list
        pass

    # 2. Convert axial to Cartesian
    # Flat-Top lattice:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Add epsilon to prevent floating point overlap detection
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = math.sqrt(3) * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # 3. Define hexagon vertices function
    def get_hex_vertices(cx, cy, side, angle_rad):
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            verts.append((vx, vy))
        return verts

    # 4. Calculate minimal outer side length S
    # Outer hexagon: center (0,0), angle 0, side S.
    # Inner hexagons: side 1, angle 0.
    
    # The condition for a point P to be inside a regular hexagon of side S, center 0, angle 0:
    # The hexagon is the intersection of 6 half-planes.
    # The normals to the sides are at angles 30, 90, 150, 210, 270, 330 degrees.
    
    normals = []
    for i in range(6):
        theta = math.pi / 6.0 + i * math.pi / 3.0 # 30, 90, 150...
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    inner_side = 1.0
    inner_angle_rad = 0.0
    
    for cx, cy in centers:
        verts = get_hex_vertices(cx, cy, inner_side, inner_angle_rad)
        for vx, vy in verts:
            for nx, ny in normals:
                proj = vx * nx + vy * ny
                if proj > max_projection:
                    max_projection = proj
                    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    S = 2.0 * max_projection / math.sqrt(3.0)
    
    # Add a tiny epsilon for floating point safety
    S += 1e-9
    
    # 5. Construct output
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0]) # angle in degrees
        
    center = (0.0, 0.0)
    side = S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
