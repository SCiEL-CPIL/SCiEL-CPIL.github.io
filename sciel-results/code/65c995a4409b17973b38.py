import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    sqrt3 = math.sqrt(3.0)
    
    inner = []
    
    # Helper to convert axial (q, r) to cartesian (x, y) for flat-top hexagons
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    # 1. Central hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. The 6 neighbors (k=2 cluster)
    neighbors = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    for q, r in neighbors:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # 3. Add 5 hexes from the next ring (k=3)
    extra_hexes = [
        (2, 0),
        (2, -1),
        (1, 1),
        (-1, 2),
        (-2, 1)
    ]
    
    for q, r in extra_hexes:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # Scale positions slightly to ensure strict disjointness (distance > sqrt(3))
    scale_factor = 1.001
    for i in range(len(inner)):
        inner[i][0] *= scale_factor
        inner[i][1] *= scale_factor

    # Calculate minimum outer side length
    # Outer hexagon: center (0,0), angle 0 (vertex at (S,0))
    # Constraints: for k in 0..5, dot(P, normal_k) <= S * sqrt(3) / 2
    # Normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Actually, if vertex is at 0, edge normals are at 30, 90...
    # Let's verify: Vertex (S,0). Edge to (S/2, S*sqrt(3)/2).
    # Midpoint of edge: (3S/4, S*sqrt(3)/4).
    # Normal vector should be perpendicular to edge.
    # Edge vector: (-S/2, S*sqrt(3)/2). Direction 120 deg.
    # Normal direction 30 deg.
    # Dot product of midpoint with normal (cos30, sin30):
    # (3S/4)(sqrt3/2) + (S*sqrt3/4)(1/2) = 3S*sqrt3/8 + S*sqrt3/8 = 4S*sqrt3/8 = S*sqrt3/2.
    # So yes, constraint is dot(P, n_k) <= S * sqrt(3) / 2.
    
    normals = []
    for k in range(6):
        angle = math.pi / 6 + k * math.pi / 3
        normals.append((math.cos(angle), math.sin(angle)))
    
    max_required_s = 0.0
    
    # Vertices of a unit flat-top hexagon relative to its center
    hex_verts = [
        (1.0, 0.0),
        (0.5, sqrt3 / 2.0),
        (-0.5, sqrt3 / 2.0),
        (-1.0, 0.0),
        (-0.5, -sqrt3 / 2.0),
        (0.5, -sqrt3 / 2.0)
    ]
    
    for cx, cy, _ in inner:
        for vx, vy in hex_verts:
            px = cx + vx
            py = cy + vy
            for nx, ny in normals:
                val = px * nx + py * ny
                # val <= S * sqrt(3) / 2  =>  S >= val * 2 / sqrt(3)
                if val > 0:
                    s_req = val * 2.0 / sqrt3
                    if s_req > max_required_s:
                        max_required_s = s_req
    
    # Add a small margin for numerical safety
    side = max_required_s * 1.001
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
