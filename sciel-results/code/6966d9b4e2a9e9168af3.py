import math
import numpy as np

N = 11

def construct():
    # Generate hexagonal lattice points
    points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            # Hex distance
            d = (abs(q) + abs(r) + abs(q+r)) / 2
            if d <= 2:
                # Convert to Cartesian
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                points.append((x, y))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take first N
    selected = points[:N]
    
    # Apply gap factor to ensure strict disjointness
    gap = 1.001
    inner_positions = []
    for p in selected:
        x, y = p[0] * gap, p[1] * gap
        inner_positions.append([x, y, 0.0])
        
    # Calculate minimal L
    # Max projection on 6 normals of the outer hexagon
    max_proj = 0.0
    for i in range(6):
        angle = i * math.pi / 3
        nx = math.cos(angle)
        ny = math.sin(angle)
        for pos in inner_positions:
            proj = pos[0] * nx + pos[1] * ny
            if proj > max_proj:
                max_proj = proj
                
    # L >= 1 + (2/sqrt(3)) * max_proj
    # The '1' comes from the apothem of the unit hexagon
    L = 1.0 + (2.0 / math.sqrt(3)) * max_proj
    
    # Add a small buffer to L to ensure validity against floating point errors
    L *= 1.001
    
    return inner_positions, (0.0, 0.0), L, 0.0
