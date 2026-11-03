import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def min_hexagon_side(vertices, cx, cy, angle_deg):
    angle = np.radians(angle_deg)
    inradius_needed = 0
    for i in range(6):
        normal_angle = angle + i * np.pi / 3 + np.pi / 6
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        for v in vertices:
            dist = (v[0] - cx) * nx + (v[1] - cy) * ny
            if dist > inradius_needed:
                inradius_needed = dist
    return inradius_needed * 2 / np.sqrt(3)

def optimize_outer_hexagon(all_vertices):
    best_side = float('inf')
    best_cx = 0
    best_cy = 0
    best_angle = 0
    
    centroid_x = np.mean(all_vertices[:, 0])
    centroid_y = np.mean(all_vertices[:, 1])
    
    for angle_deg in np.arange(0, 60, 0.5):
        for cx_offset in np.arange(-1.0, 1.01, 0.25):
            for cy_offset in np.arange(-1.0, 1.01, 0.25):
                cx = centroid_x + cx_offset
                cy = centroid_y + cy_offset
                side = min_hexagon_side(all_vertices, cx, cy, angle_deg)
                if side < best_side:
                    best_side = side
                    best_cx = cx
                    best_cy = cy
                    best_angle = angle_deg
    
    return best_cx, best_cy, best_side, best_angle

def generate_cluster_shape(shape_type):
    shapes = {
        '3431': [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3)],
        'flower': [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (0,2)],
        'compact': [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,-1), (-2,1)],
        'diamond': [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (2,0), (-2,0), (0,2), (0,-2)],
        'row': [(i, 0) for i in range(11)],
        'hex7': [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,1), (-2,-1)],
        'hex7b': [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (0,2), (0,-2)],
    }
    
    coords = shapes.get(shape_type, shapes['3431'])
    
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
        positions.append((x, y))
    
    return positions

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    shape_types = ['3431', 'flower', 'compact', 'diamond', 'row', 'hex7', 'hex7b']
    
    for shape_type in shape_types:
        if time.time() - start_time > 100:
            break
        
        positions = generate_cluster_shape(shape_type)
        
        for inner_angle in [0.0, 30.0]:
            if time.time() - start_time > 100:
                break
            
            all_vertices = []
            inner = []
            for i, (px, py) in enumerate(positions):
                verts = hex_vertices(px, py, 1.0, inner_angle)
                all_vertices.append(verts)
                inner.append([px, py, inner_angle])
            
            all_vertices = np.vstack(all_vertices)
            
            cx, cy, side, angle_deg = optimize_outer_hexagon(all_vertices)
            
            if side < best_score:
                best_score = side
                best_result = (inner, (cx, cy), side, angle_deg)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
