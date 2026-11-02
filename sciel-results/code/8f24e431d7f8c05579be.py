import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 centers
    centers = []
    # Core 7
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = k * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
    
    # 4 Extras at specific axial positions converted to Cartesian
    # Axial (1,1) -> (1.5*sqrt(3), 1.5)
    # Axial (-1,-1) -> (-1.5*sqrt(3), -1.5)
    # Axial (-1,2) -> (-0.5*sqrt(3), 3.0)
    # Axial (1,-2) -> (0.5*sqrt(3), -3.0)
    
    extras = [
        (1.5 * math.sqrt(3), 1.5),
        (-1.5 * math.sqrt(3), -1.5),
        (-0.5 * math.sqrt(3), 3.0),
        (0.5 * math.sqrt(3), -3.0)
    ]
    centers.extend(extras)
    
    # Calculate required outer side length S
    # Outer hexagon aligned with lattice (angle 0)
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    max_S = 0.0
    for k in range(6):
        phi = k * math.pi / 3
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        # Support function of unit hexagon in direction phi
        # Vertices of unit hexagon
        verts = [
            (1.0, 0.0),
            (0.5, math.sqrt(3)/2),
            (-0.5, math.sqrt(3)/2),
            (-1.0, 0.0),
            (-0.5, -math.sqrt(3)/2),
            (0.5, -math.sqrt(3)/2)
        ]
        R_phi = max(v[0]*cos_phi + v[1]*sin_phi for v in verts)
        
        # Max projection of centers
        max_proj = 0.0
        for (cx, cy) in centers:
            proj = cx * cos_phi + cy * sin_phi
            if proj > max_proj:
                max_proj = proj
        
        # Required apothem
        apothem = max_proj + R_phi
        # S * sqrt(3)/2 >= apothem
        S_req = apothem * 2 / math.sqrt(3)
        if S_req > max_S:
            max_S = S_req
            
    # Add a small safety margin
    S = max_S + 0.01
    
    # Format inner list
    inner = []
    for (x, y) in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
