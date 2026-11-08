import numpy as np

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return vertices

def construct():
    sqrt3 = np.sqrt(3)
    
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3
            if row % 2 == 1:
                x += sqrt3 / 2
            y = row * 1.5
            positions.append((x, y, 0.0))
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_offset = (min(xs) + max(xs)) / 2
    cy_offset = (min(ys) + max(ys)) / 2
    
    inner = [[p[0] - cx_offset, p[1] - cy_offset, p[2]] for p in positions]
    
    all_verts = []
    for ix, iy, iangle in inner:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    best_side = float('inf')
    best_center = (float(centroid[0]), float(centroid[1]))
    best_angle = 0.0
    
    for angle_deg in np.linspace(0, 60, 721):
        normal_angles = np.radians(angle_deg + 30 + np.array([0, 60, 120, 180, 240, 300]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        rel_verts = all_verts - centroid
        max_projs = np.max(rel_verts @ normals.T, axis=0)
        
        apothem_needed = np.max(max_projs)
        side_needed = 2 * apothem_needed / sqrt3
        
        if side_needed < best_side:
            best_side = side_needed
            best_angle = float(angle_deg)
            best_center = (float(centroid[0]), float(centroid[1]))
    
    # Add tiny margin for safety
    best_side += 1e-9
    
    return inner, best_center, best_side, best_angle
