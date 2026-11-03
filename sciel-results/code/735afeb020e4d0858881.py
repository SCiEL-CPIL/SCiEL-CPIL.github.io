import numpy as np
import math

N = 12

def _hex_vertices(center, side=1.0, angle=0.0):
    """
    Generate vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle: rotation in degrees
    """
    cx, cy = center
    rad = math.radians(angle)
    verts = []
    for i in range(6):
        theta = rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def construct():
    # Define the 12 hexagons in axial coordinates
    # Axial to Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1)
    ]
    
    inner = []
    all_vertices = []
    
    for q, r in coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        # Inner hexagons are aligned with the grid (angle 0)
        inner.append([x, y, 0.0])
        
        # Get vertices
        verts = _hex_vertices((x, y), 1.0, 0.0)
        all_vertices.extend(verts)
        
    # Find centroid of all vertices
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    # Translate vertices to be centered at origin for optimization
    centered_vertices = [(x - cx, y - cy) for x, y in all_vertices]
    
    # Optimize rotation of the outer hexagon
    # The outer hexagon has normals at angles theta, theta+60, theta+120
    # The constraint is max projection on normals <= S * sqrt(3) / 2
    # So S(theta) = (2/sqrt(3)) * max_k | v . n_k(theta) |
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Sample theta from 0 to 60 degrees
    # Use finer steps for better precision
    for i in range(1000):
        theta = i * 60.0 / 1000.0
        rad = math.radians(theta)
        
        # Normals
        n0 = (math.cos(rad), math.sin(rad))
        n1 = (math.cos(rad + 2*math.pi/3), math.sin(rad + 2*math.pi/3))
        n2 = (math.cos(rad + 4*math.pi/3), math.sin(rad + 4*math.pi/3))
        
        max_proj = 0.0
        for vx, vy in centered_vertices:
            p0 = abs(vx * n0[0] + vy * n0[1])
            p1 = abs(vx * n1[0] + vy * n1[1])
            p2 = abs(vx * n2[0] + vy * n2[1])
            max_proj = max(max_proj, p0, p1, p2)
            
        S = (2.0 / math.sqrt(3.0)) * max_proj
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Add a tiny epsilon to avoid boundary issues
    best_S += 1e-9
    
    # The outer hexagon center is the centroid (cx, cy)
    # The outer hexagon angle is best_theta
    # The inner hexagons are as defined (angle 0)
    
    return inner, (cx, cy), best_S, best_theta
