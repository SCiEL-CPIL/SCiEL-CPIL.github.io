import numpy as np

N = 11

def construct():
    sqrt3 = np.sqrt(3)
    
    # Generate hexagonal cluster positions
    positions = []
    
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt3
    for k in range(6):
        angle = k * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append((x, y))
    
    # Ring 2: add 4 more from hexagonal lattice
    ring2_axial = [(2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (-1,-2), (-2,-1)]
    
    for (q, r) in ring2_axial[:4]:
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        positions.append((x, y))
    
    positions = positions[:N]
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Find minimum enclosing regular hexagon
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (px, py) in positions:
        for v in range(6):
            va = v * np.pi / 3
            vx = px + np.cos(va)
            vy = py + np.sin(va)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    best_s = float('inf')
    best_angle = 0.0
    
    for theta_deg in np.arange(0, 60, 2):
        theta = np.radians(theta_deg)
        max_proj = 0
        for k in range(6):
            a = theta + k * np.pi / 3
            nx = np.cos(a)
            ny = np.sin(a)
            projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs))
        
        s = 2 * max_proj / sqrt3
        if s < best_s:
            best_s = s
            best_angle = theta_deg
    
    return inner, (cx, cy), best_s, best_angle
