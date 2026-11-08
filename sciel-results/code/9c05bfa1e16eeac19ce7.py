import numpy as np
import time
import math

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def project_points(points, axis):
    min_proj = float('inf')
    max_proj = float('-inf')
    ax, ay = axis
    for p in points:
        proj = p[0]*ax + p[1]*ay
        if proj < min_proj: min_proj = proj
        if proj > max_proj: max_proj = proj
    return min_proj, max_proj

def polygons_overlap(p1, p2):
    axes = []
    # Edges of p1
    for i in range(6):
        j = (i+1) % 6
        ex = p1[i][0] - p1[j][0]
        ey = p1[i][1] - p1[j][1]
        nx = -ey
        ny = ex
        norm = math.sqrt(nx*nx + ny*ny)
        if norm > 1e-10:
            axes.append((nx/norm, ny/norm))
            
    # Edges of p2
    for i in range(6):
        j = (i+1) % 6
        ex = p2[i][0] - p2[j][0]
        ey = p2[i][1] - p2[j][1]
        nx = -ey
        ny = ex
        norm = math.sqrt(nx*nx + ny*ny)
        if norm > 1e-10:
            axes.append((nx/norm, ny/norm))
            
    for axis in axes:
        min1, max1 = project_points(p1, axis)
        min2, max2 = project_points(p2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_overlap(positions, angles):
    n = len(positions)
    verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polygons_overlap(verts_list[i], verts_list[j]):
                return True
    return False

def calculate_required_L(positions, angles):
    max_proj = 0
    normals = []
    for i in range(6):
        a = i * math.pi / 3
        normals.append((math.cos(a), math.sin(a)))
        
    for i in range(len(positions)):
        verts = hex_vertices(positions[i][0], positions[i][1], angles[i])
        for v in verts:
            for n in normals:
                proj = abs(v[0]*n[0] + v[1]*n[1])
                if proj > max_proj:
                    max_proj = proj
    return max_proj * 2.0 / math.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration
    positions = [(0.0, 0.0)]
    angles = [0.0]
    
    # Ring 1
    for i in range(6):
        a = i * 60.0
        rad = math.radians(a)
        x = 2.0 * math.cos(rad)
        y = 2.0 * math.sin(rad)
        positions.append((x, y))
        angles.append(0.0)
        
    # Extra 4
    for i in range(4):
        a = 30.0 + i * 60.0
        rad = math.radians(a)
        x = 3.5 * math.cos(rad)
        y = 3.5 * math.sin(rad)
        positions.append((x, y))
        angles.append(0.0)
        
    current_L = calculate_required_L(positions, angles)
    best_L = current_L
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    
    T = 1.0
    cooling = 0.9995
    step_size = 0.5
    
    while time.time() < deadline:
        i = np.random.randint(0, N)
        
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        dangle = np.random.normal(0, 10.0)
        
        old_x, old_y = positions[i]
        old_angle = angles[i]
        
        positions[i] = (old_x + dx, old_y + dy)
        angles[i] = old_angle + dangle
        
        if check_overlap(positions, angles):
            positions[i] = (old_x, old_y)
            angles[i] = old_angle
            continue
            
        new_L = calculate_required_L(positions, angles)
        
        if new_L < current_L:
            current_L = new_L
            if new_L < best_L:
                best_L = new_L
                best_positions = [p[:] for p in positions]
                best_angles = angles[:]
        else:
            delta = new_L - current_L
            if delta > 0:
                prob = math.exp(-delta / T)
                if np.random.random() < prob:
                    current_L = new_L
                else:
                    positions[i] = (old_x, old_y)
                    angles[i] = old_angle
            else:
                current_L = new_L
                if new_L < best_L:
                    best_L = new_L
                    best_positions = [p[:] for p in positions]
                    best_angles = angles[:]
                    
        T *= cooling
        
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
