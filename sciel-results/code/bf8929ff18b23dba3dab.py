import numpy as np
from scipy.spatial import ConvexHull

def construct():
    def hex_vertices(cx, cy, angle_deg=0.0):
        theta = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            a = theta + k * np.pi / 3
            vertices.append((cx + np.cos(a), cy + np.sin(a)))
        return vertices
    
    sqrt3 = np.sqrt(3)
    
    positions = []
    for i in range(3):
        positions.append((i * sqrt3, 0.0))
    for i in range(5):
        positions.append((sqrt3/2 + i * sqrt3, 1.5))
    for i in range(3):
        positions.append((i * sqrt3, 3.0))
    
    all_vertices = []
    for (cx, cy) in positions:
        verts = hex_vertices(cx, cy, 0.0)
        all_vertices.extend(verts)
    
    all_vertices = np.array(all_vertices)
    hull = ConvexHull(all_vertices)
    hull_vertices = all_vertices[hull.vertices]
    
    centroid = np.mean(hull_vertices, axis=0)
    
    best_L = float('inf')
    best_theta = 0.0
    best_center = centroid
    
    for theta_deg in np.linspace(0, 60, 601):
        theta = np.radians(theta_deg)
        normals = np.array([(np.cos(theta + k * np.pi / 3 + np.pi / 6), 
                            np.sin(theta + k * np.pi / 3 + np.pi / 6)) for k in range(6)])
        
        center = centroid
        rel = hull_vertices - center
        projections = rel @ normals.T
        max_proj = np.max(projections, axis=0)
        L = np.max(max_proj) / (sqrt3 / 2)
        if L < best_L:
            best_L = L
            best_theta = theta_deg
            best_center = center
    
    # Refine
    for theta_deg in np.linspace(best_theta - 3, best_theta + 3, 301):
        theta = np.radians(theta_deg)
        normals = np.array([(np.cos(theta + k * np.pi / 3 + np.pi / 6), 
                            np.sin(theta + k * np.pi / 3 + np.pi / 6)) for k in range(6)])
        
        for dx in np.linspace(-1, 1, 101):
            for dy in np.linspace(-1, 1, 101):
                center = centroid + np.array([dx, dy])
                rel = hull_vertices - center
                projections = rel @ normals.T
                max_proj = np.max(projections, axis=0)
                L = np.max(max_proj) / (sqrt3 / 2)
                if L < best_L:
                    best_L = L
                    best_theta = theta_deg
                    best_center = center
    
    inner = [[pos[0], pos[1], 0.0] for pos in positions]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_L), float(best_theta)
