import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Place 12 unit hexagons in a compact hexagonal cluster
    hex_positions_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2),
    ]
    
    positions = []
    for q, r in hex_positions_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for x, y, angle in inner:
        for k in range(6):
            theta = angle + k * np.pi / 3.0
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            all_vertices.append((vx, vy))
    
    points = np.array(all_vertices)
    
    def hexagon_side_length(points, center, theta):
        cx, cy = center
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            normals.append((np.cos(angle), np.sin(angle)))
        
        max_extent = 0.0
        for px, py in points:
            for nx, ny in normals:
                d = (px - cx) * nx + (py - cy) * ny
                if d > max_extent:
                    max_extent = d
        
        return 2.0 * max_extent / np.sqrt(3.0)
    
    def obj(center, theta, points):
        return hexagon_side_length(points, center, theta)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_theta = 0.0
    
    thetas = np.linspace(0, np.pi/3, 61)
    
    for theta in thetas:
        pts = points
        x0 = [np.mean(pts[:, 0]), np.mean(pts[:, 1])]
        
        res = minimize(obj, x0, args=(theta, points), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if res.fun < best_side:
            best_side = res.fun
            best_center = (res.x[0], res.x[1])
            best_theta = theta
    
    return inner, best_center, best_side, np.degrees(best_theta)
