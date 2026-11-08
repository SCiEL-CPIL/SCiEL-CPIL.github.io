import numpy as np
import random
import math
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Vertices of a unit regular hexagon centered at origin
    angles = np.array([k * np.pi / 3 for k in range(6)])
    base_vertices = np.array([np.cos(angles), np.sin(angles)]).T
    
    def get_all_points(positions, rotations):
        points = []
        for i in range(N):
            x, y = positions[i]
            phi = rotations[i]
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            verts = R @ base_vertices.T + np.array([x, y])
            points.append(verts.T)
        return np.vstack(points)

    def estimate_min_side(points):
        min_side = float('inf')
        for theta in np.linspace(0, np.pi/6, 7):
            normals = []
            for k in range(6):
                ang = theta + k * np.pi / 3
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)
            
            projections = points @ normals.T
            mins = np.min(projections, axis=0)
            maxs = np.max(projections, axis=0)
            
            widths = maxs - mins
            max_width = np.max(widths)
            S = max_width * 2 / np.sqrt(3)
            
            if S < min_side:
                min_side = S
        return min_side

    # Initialize positions and rotations
    positions = []
    rotations = []
    
    # 1 center
    positions.append([0.0, 0.0])
    rotations.append(0.0)
    
    # 6 around it
    for k in range(6):
        ang = k * np.pi / 3
        dist = np.sqrt(3)
        x = dist * np.cos(ang)
        y = dist * np.sin(ang)
        positions.append([x, y])
        rotations.append(0.0)
        
    # 5 more
    for k in range(5):
        ang = k * np.pi / 3 + np.pi/6
        dist = 2 * np.sqrt(3)
        x = dist * np.cos(ang)
        y = dist * np.sin(ang)
        positions.append([x, y])
        rotations.append(0.0)
        
    best_S = estimate_min_side(get_all_points(positions, rotations))
    current_positions = [p[:] for p in positions]
    current_rotations = rotations[:]
    
    for i in range(20000):
        idx = random.randint(0, N-1)
        
        new_pos = current_positions[idx][:]
        new_rot = current_rotations[idx]
        
        if random.random() < 0.5:
            new_pos[0] += random.gauss(0, 0.5)
            new_pos[1] += random.gauss(0, 0.5)
        else:
            new_rot += random.gauss(0, 0.2)
            
        old_pos = current_positions[idx]
        old_rot = current_rotations[idx]
        
        current_positions[idx] = new_pos
        current_rotations[idx] = new_rot
        
        points = get_all_points(current_positions, current_rotations)
        S = estimate_min_side(points)
        
        if S < best_S:
            best_S = S
        else:
            current_positions[idx] = old_pos
            current_rotations[idx] = old_rot
            
    points = get_all_points(current_positions, current_rotations)
    
    best_S_final = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for theta in np.linspace(0, np.pi/6, 37):
        normals = []
        for k in range(6):
            ang = theta + k * np.pi / 3
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        projections = points @ normals.T
        mins = np.min(projections, axis=0)
        maxs = np.max(projections, axis=0)
        
        h_low = np.max(maxs - mins) / 2
        h_high = 50.0
        
        for _ in range(20):
            h_mid = (h_low + h_high) / 2
            
            c_obj = np.zeros(2)
            A_ub = []
            b_ub = []
            
            for k in range(6):
                A_ub.append(normals[k])
                b_ub.append(h_mid - mins[k])
                A_ub.append(-normals[k])
                b_ub.append(h_mid - maxs[k])
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
            if res.success:
                h_high = h_mid
            else:
                h_low = h_mid
        
        S = h_high * 2 / np.sqrt(3)
        if S < best_S_final:
            best_S_final = S
            best_angle = theta
            # Recover center
            c_obj = np.zeros(2)
            A_ub = []
            b_ub = []
            for k in range(6):
                A_ub.append(normals[k])
                b_ub.append(h_high - mins[k])
                A_ub.append(-normals[k])
                b_ub.append(h_high - maxs[k])
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
            if res.success:
                best_center = tuple(res.x)

    inner = []
    for i in range(N):
        x, y = current_positions[i]
        angle_deg = np.degrees(current_rotations[i])
        inner.append([x, y, angle_deg])
        
    return inner, best_center, best_S_final, np.degrees(best_angle)
# EVOLVE-BLOCK-END
