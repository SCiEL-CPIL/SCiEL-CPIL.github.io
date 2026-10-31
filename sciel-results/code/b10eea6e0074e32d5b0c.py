import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
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
    # Pointy-top lattice:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This corresponds to hexagons with "pointy" top/bottom?
    # Let's check the orientation of the hexagons themselves.
    # If the lattice is pointy-top, the hexagons that tile it are pointy-top.
    # A pointy-top hexagon has a vertex at 90 degrees.
    # So angle = 90 degrees?
    # Or if we use flat-top lattice, angle = 0.
    
    # Let's stick to Flat-Top for simplicity, as "angle 0" often implies flat-top in many contexts (vertex at 0).
    # Flat-top lattice axial to pixel:
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    # Distance (0,0) to (1,0): x=1.5, y=sqrt(3)/2 -> dist = sqrt(2.25+0.75)=sqrt(3).
    # Distance (0,0) to (0,1): x=0, y=sqrt(3) -> dist=sqrt(3).
    # Distance (0,0) to (1,1): x=1.5, y=sqrt(3)*1.5 = 2.598 -> dist = sqrt(2.25+6.75)=3.
    # This works.
    
    # So, use Flat-Top lattice.
    # The hexagons in a flat-top tiling are flat-top.
    # Angle = 0.
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
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
    # (Since vertices are at 0, 60, ..., the sides are perpendicular to these angles).
    # Wait.
    # Vertex at 0: (S, 0).
    # Vertex at 60: (S/2, S*sqrt(3)/2).
    # Side between 0 and 60:
    # Midpoint: (3S/4, S*sqrt(3)/4).
    # Normal angle: 30 degrees.
    # Distance from origin to this side (apothem): S * sqrt(3) / 2.
    # So the constraint is: P . n <= S * sqrt(3) / 2 for all 6 normals n.
    
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
