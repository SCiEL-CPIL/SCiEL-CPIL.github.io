import numpy as np
import math
import random

N = 11

def get_hexagon_vertices_np(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    angles = angle_rad + np.arange(6) * math.pi / 3
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def polygons_overlap_np(v1, v2):
    def get_axes(vertices):
        edges = np.roll(vertices, -1, axis=0) - vertices
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals = normals / norms
        return normals
    
    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    axes = np.vstack([axes1, axes2])
    
    proj1 = v1 @ axes.T
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    
    proj2 = v2 @ axes.T
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    if np.any(max1 < min2) or np.any(max2 < min1):
        return False
    return True

def calculate_required_side_length_np(hexagons_verts):
    # hexagons_verts: list of (6,2) arrays
    max_proj = 0.0
    # Normals for flat-top outer hexagon
    angles = np.arange(6) * math.pi / 3
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    for verts in hexagons_verts:
        # Project all vertices of this hexagon onto all 6 normals
        projs = verts @ normals.T
        max_in_hex = np.max(projs)
        if max_in_hex > max_proj:
            max_proj = max_in_hex
            
    return max_proj

def construct():
    random.seed(42)
    
    hexagons = []
    for i in range(N):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        ang = random.uniform(0, 360)
        hexagons.append([x, y, ang])
        
    best_hexagons = [h[:] for h in hexagons]
    best_cost = float('inf')
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    iterations = 3000 # Reduced to ensure speed
    
    for it in range(iterations):
        hex_verts = [get_hexagon_vertices_np(h[0], h[1], h[2]) for h in hexagons]
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap_np(hex_verts[i], hex_verts[j]):
                    overlap_penalty += 100.0
                    
        side_len = calculate_required_side_length_np(hex_verts)
        current_cost = side_len + overlap_penalty
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_hexagons = [h[:] for h in hexagons]
            
        idx = random.randint(0, N-1)
        delta_x = random.gauss(0, 0.5)
        delta_y = random.gauss(0, 0.5)
        delta_ang = random.gauss(0, 10)
        
        new_hexagons = [h[:] for h in hexagons]
        new_hexagons[idx][0] += delta_x
        new_hexagons[idx][1] += delta_y
        new_hexagons[idx][2] += delta_ang
        
        new_hex_verts = [get_hexagon_vertices_np(h[0], h[1], h[2]) for h in new_hexagons]
        new_overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap_np(new_hex_verts[i], new_hex_verts[j]):
                    new_overlap_penalty += 100.0
                    
        new_side_len = calculate_required_side_length_np(new_hex_verts)
        new_cost = new_side_len + new_overlap_penalty
        
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            hexagons = new_hexagons
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final check and fallback
    final_hex_verts = [get_hexagon_vertices_np(h[0], h[1], h[2]) for h in best_hexagons]
    has_overlap = False
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_np(final_hex_verts[i], final_hex_verts[j]):
                has_overlap = True
                break
        if has_overlap:
            break
            
    if has_overlap:
        inner_fallback = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        fallback_side = (N-1)*math.sqrt(3)/2 + 1
        return inner_fallback, (0.0, 0.0), fallback_side, 0.0

    final_side_len = calculate_required_side_length_np(final_hex_verts)
    return best_hexagons, (0.0, 0.0), final_side_len, 0.0
