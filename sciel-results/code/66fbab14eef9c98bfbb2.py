"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    SIDE_INNER = 1.0
    CIRCUMRADIUS_INNER = SIDE_INNER
    OUTER_INRADIUS_FACTOR = math.sqrt(3) / 2.0

    def get_hex_vertices(cx, cy, angle_deg):
        rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = rad + i * math.pi / 3
            x = cx + CIRCUMRADIUS_INNER * math.cos(theta)
            y = cy + CIRCUMRADIUS_INNER * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def project_polygon(vertices, axis):
        dots = np.dot(vertices, axis)
        return np.min(dots), np.max(dots)

    def min_penetration_depth(poly1, poly2):
        axes = []
        for i in range(len(poly1)):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % len(poly1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(len(poly2)):
            p1 = poly2[i]
            p2 = poly2[(i + 1) % len(poly2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        min_overlap = float('inf')
        for axis in axes:
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return 0.0
            else:
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_overlap:
                    min_overlap = overlap
        return min_overlap if min_overlap < float('inf') else 0.0

    def calculate_required_S(positions, angles):
        max_proj = 0.0
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
        for i in range(N):
            cx, cy = positions[i]
            angle = angles[i]
            verts = get_hex_vertices(cx, cy, angle)
            for v in verts:
                for nx, ny in normals:
                    proj = v[0] * nx + v[1] * ny
                    if proj > max_proj:
                        max_proj = proj
                        
        return max_proj / OUTER_INRADIUS_FACTOR

    def objective(params):
        positions = []
        angles = []
        for i in range(N):
            positions.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
            
        S_req = calculate_required_S(positions, angles)
        
        penalty = 0.0
        verts_list = [get_hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        
        for i in range(N):
            for j in range(i+1, N):
                depth = min_penetration_depth(verts_list[i], verts_list[j])
                if depth > 0:
                    penalty += depth * depth
                    
        return S_req + 1000.0 * penalty

    # Initial guess
    dist = math.sqrt(3)
    init_params = [0.0, 0.0, 0.0]
    for i in range(6):
        theta = math.radians(i * 60)
        init_params.extend([dist * math.cos(theta), dist * math.sin(theta), 0.0])
        
    lattice_points = [(2, 0), (0, 2), (-2, 0), (1, 1), (-1, -1)]
    for li, lj in lattice_points:
        x = li * dist + lj * (dist / 2.0)
        y = lj * (dist * math.sqrt(3) / 2.0)
        init_params.extend([x, y, 0.0])
        
    best_params = init_params
    best_score = objective(init_params)
    
    try:
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'fatol': 1e-5, 'xatol': 1e-5})
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass

    # Extract results
    final_positions = []
    final_angles = []
    for i in range(N):
        final_positions.append((best_params[3*i], best_params[3*i+1]))
        final_angles.append(best_params[3*i+2])
        
    S_final = calculate_required_S(final_positions, final_angles)
    
    # Validity check
    verts_list = [get_hex_vertices(p[0], p[1], a) for p, a in zip(final_positions, final_angles)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if min_penetration_depth(verts_list[i], verts_list[j]) > 1e-4:
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
        
    return inner, (0.0, 0.0), S_final, 0.0
# EVOLVE-BLOCK-END
