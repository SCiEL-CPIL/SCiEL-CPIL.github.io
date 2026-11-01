import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        # Normals for flat-top hexagon (angle 0) are at 30, 90, 150, 210, 270, 330
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                return True
        return False

    def check_validity(inner_positions, outer_side, outer_angle=0.0):
        # Outer hexagon centered at (0,0)
        for i in range(N):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, 0.0, 0.0, outer_side, outer_angle):
                    return False
        
        for i in range(N):
            for j in range(i+1, N):
                ci, ai = (inner_positions[i][0], inner_positions[i][1]), inner_positions[i][2]
                cj, aj = (inner_positions[j][0], inner_positions[j][1]), inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def objective(params):
        inner_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        outer_side = params[3*N]
        if outer_side < 0.1:
            return 1000.0
            
        # Check validity
        valid = True
        penalty = 0.0
        
        # Inside check
        for i in range(N):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, 0.0, 0.0, outer_side, 0.0):
                    valid = False
                    penalty += 1.0
                    break # One violation per hexagon is enough for penalty
        
        # Disjoint check
        for i in range(N):
            for j in range(i+1, N):
                ci, ai = (inner_positions[i][0], inner_positions[i][1]), inner_positions[i][2]
                cj, aj = (inner_positions[j][0], inner_positions[j][1]), inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    valid = False
                    penalty += 1.0
        
        if valid:
            return outer_side
        else:
            return outer_side + penalty * 10.0

    # Initial configuration: Compact cluster
    # Axial coordinates
    axial = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
        (-2, 3), (-1, 3)
    ]
    
    # Convert to Cartesian
    initial_positions = []
    for q, r in axial:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        initial_positions.append([x, y, 0.0])
        
    # Center the cluster
    xs = [p[0] for p in initial_positions]
    ys = [p[1] for p in initial_positions]
    cx_mean = np.mean(xs)
    cy_mean = np.mean(ys)
    for p in initial_positions:
        p[0] -= cx_mean
        p[1] -= cy_mean

    # Estimate initial outer side
    # Max distance from center to any vertex
    max_dist = 0
    for p in initial_positions:
        verts = hexagon_vertices(p[0], p[1], 1.0, p[2])
        for v in verts:
            d = np.linalg.norm(v)
            if d > max_dist:
                max_dist = d
    # Hexagon of side S has apothem S*sqrt(3)/2.
    # We need the hexagon to contain points at distance max_dist.
    # The "radius" of a hexagon varies between apothem and side.
    # To be safe, assume we need S such that apothem >= max_dist / (something).
    # Actually, just start with a generous S and let it shrink.
    initial_outer_side = max_dist / (np.sqrt(3)/2) * 1.1
    
    params = []
    for p in initial_positions:
        params.extend([p[0], p[1], p[2]])
    params.append(initial_outer_side)
    
    best_params = params.copy()
    best_score = objective(params)
    
    # Optimize
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    final_params = best_params
    
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append([x, y, a])
    
    os = max(final_params[3*N], 0.1)
    
    # Final validation
    if not check_validity(inner, os, 0.0):
        # Fallback to initial
        inner = [p.copy() for p in initial_positions]
        os = initial_outer_side
        # Ensure fallback is valid, expand if needed
        while not check_validity(inner, os, 0.0) and os < 10:
            os += 0.1

    return inner, (0.0, 0.0), os, 0.0

