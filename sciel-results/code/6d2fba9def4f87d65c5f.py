import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute unit hexagon vertices for angle 0 (flat-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        min_gap = float('inf')
        
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def compute_overlap_penalty(positions, angles, weight=1000.0):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += weight * gap**2
        return penalty

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        side_len = calc_min_outer_side(positions, angles, 0.0)
        pen = compute_overlap_penalty(positions, angles)
        return side_len + pen

    # Generate initial configuration using hexagonal lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Use Nelder-Mead for local optimization
    # It handles non-smooth functions (like the penalty) reasonably well
    options = {
        'maxiter': 1000,
        'xatol': 1e-6,
        'fatol': 1e-6
    }
    
    # Try multiple restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Perturb initial point for restarts > 0
        if r == 0:
            current_x = x0.copy()
        else:
            current_x = x0 + np.random.normal(0, 0.5, size=x0.shape)
            
        res = minimize(objective, current_x, method='Nelder-Mead', options=options)
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    # Final check and refinement
    # If the best solution has overlap, try to push them apart with a few gradient steps or just return it if it's "close"
    # The penalty should have minimized overlap.
    
    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    
    # Verify validity
    polys = [get_hex_verts(final_positions[i][0], final_positions[i][1], 1.0, final_angles[i]) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if sat_gap(polys[i], polys[j]) < -1e-5:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to initial lattice if optimization failed badly
        final_positions = init_positions
        final_angles = init_angles
        
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
