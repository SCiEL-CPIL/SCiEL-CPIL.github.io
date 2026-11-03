import numpy as np

N = 11

def construct():
    # Generate inner hexagons
    centers = []
    centers.append((0, 0))
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    centers.extend(ring1)
    ring2_subset = [(2, 0), (1, 1), (0, 2), (-1, 2)]
    centers.extend(ring2_subset)
    
    inner = []
    vertices = []
    for q, r in centers:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        inner.append([x, y, 0.0])
        
        # Add vertices of this hexagon to the list of points to check
        # Vertices of a unit hexagon centered at (x,y) with angle 0
        for k in range(6):
            theta = np.pi / 3 * k
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            vertices.append((vx, vy))
            
    vertices = np.array(vertices)
    
    # Search for best outer angle
    best_s = float('inf')
    best_angle = 0.0
    
    # Check angles 0 to 30 degrees
    for deg in np.linspace(0, 30, 31):
        phi = np.deg2rad(deg)
        max_proj = 0
        for k in range(3):
            angle = phi + k * np.pi / 3
            proj = vertices[:, 0] * np.cos(angle) + vertices[:, 1] * np.sin(angle)
            max_proj = max(max_proj, np.max(np.abs(proj)))
            
        s = 2.0 / np.sqrt(3) * max_proj
        if s < best_s:
            best_s = s
            best_angle = deg
            
    # Add a small margin for safety
    outer_side = best_s * 1.001
    outer_center = (0.0, 0.0)
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
