"""Pack 11 unit hexagons into minimal regular hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2  # inradius of unit hexagon
    
    # Pointy-top hexagon vertices (circumradius 1, inradius sqrt(3)/2)
    # Vertices at 30, 90, 150, 210, 270, 330 degrees
    hex_angle_base = math.radians(30)
    hex_verts = [(math.cos(hex_angle_base + i * math.pi / 3), 
                  math.sin(hex_angle_base + i * math.pi / 3)) for i in range(6)]
    
    # Compact 4-3-4 arrangement of hexagon centers
    positions = []
    # Row 0: 4 hexagons at y=0
    for i in range(4):
        positions.append((i * sqrt3, 0.0))
    # Row 1: 3 hexagons at y=1.5, offset by sqrt3/2
    for i in range(3):
        positions.append((sqrt3/2 + i * sqrt3, 1.5))
    # Row 2: 4 hexagons at y=3.0
    for i in range(4):
        positions.append((i * sqrt3, 3.0))
    
    hex_angle_deg = 30.0  # pointy-top orientation
    
    inner = [[px, py, hex_angle_deg] for (px, py) in positions]
    
    # Compute all vertices of all inner hexagons
    all_verts = []
    for (px, py) in positions:
        for (vx, vy) in hex_verts:
            all_verts.append((px + vx, py + vy))
    
    # Center of the cluster
    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)
    
    # For a regular hexagon with center (cx,cy), side length s, and rotation phi:
    # The 6 outward normal directions are at angles phi + k*60 degrees
    # A point (x,y) is inside if for all k:
    #   (x-cx)*cos(alpha_k) + (y-cy)*sin(alpha_k) <= s * sqrt(3)/2
    # where alpha_k = phi + k*pi/3
    
    # Minimum s for a given phi:
    # s = max_k [ max_p ((p_x - cx)*cos(alpha_k) + (p_y - cy)*sin(alpha_k)) ] / (sqrt3/2)
    
    def required_s(phi_deg, cx, cy, verts):
        phi = math.radians(phi_deg)
        max_proj = 0.0
        for k in range(6):
            alpha = phi + k * math.pi / 3
            nx = math.cos(alpha)
            ny = math.sin(alpha)
            for (px, py) in verts:
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2)
    
    # Search over outer hexagon rotation
    best_s = float('inf')
    best_phi = 0.0
    
    # Coarse search
    for phi_deg in range(0, 60, 2):
        if time.time() > deadline:
            break
        s = required_s(phi_deg, cx, cy, all_verts)
        if s < best_s:
            best_s = s
            best_phi = phi_deg
    
    # Fine search around best
    for phi_deg in [best_phi + d for d in [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]]:
        if time.time() > deadline:
            break
        s = required_s(phi_deg, cx, cy, all_verts)
        if s < best_s:
            best_s = s
            best_phi = phi_deg
    
    # The outer hexagon's "rotation" angle: if normals are at best_phi + k*60,
    # vertices are at best_phi + 30 + k*60
    # Report the vertex angle as the rotation
    outer_angle = best_phi + 30.0
    
    return inner, (cx, cy), best_s, outer_angle
# EVOLVE-BLOCK-END
