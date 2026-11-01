import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    offsets = base_angles + angle_rad
    xs = np.cos(offsets)
    ys = np.sin(offsets)
    return np.column_stack((xs + cx, ys + cy))

def hexes_overlap_penalty(v1, v2):
    # SAT check for two convex polygons
    # v1, v2 are (6,2) arrays
    def get_axes(v):
        axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    total_overlap = 0.0
    separated = False
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            separated = True
            break
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > total_overlap:
            total_overlap = overlap
            
    if separated:
        return 0.0
    else:
        return total_overlap

def calculate_L(verts, outer_angle_deg):
    outer_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_rad), np.sin(outer_rad)],
        [np.cos(outer_rad + np.pi/3), np.sin(outer_rad + np.pi/3)],
        [np.cos(outer_rad + 2*np.pi/3), np.sin(outer_rad + 2*np.pi/3)]
    ])
    
    max_proj = 0.0
    for v in verts:
        dots = np.abs(np.dot(normals, v.T)) 
        max_proj = max(max_proj, np.max(dots))
        
    L_min = (2.0 / np.sqrt(3)) * max_proj
    return L_min

def objective(params):
    inner_params = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    verts = []
    for i in range(11):
        x, y, a = inner_params[i]
        verts.append(hex_vertices(x, y, a))
        
    L_min = calculate_L(verts, outer_angle)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            ov = hexes_overlap_penalty(verts[i], verts[j])
            if ov > 0:
                penalty += ov * 1000.0 # Heavy penalty
                
    return L_min + penalty

def is_valid(inner_params, outer_angle, L):
    verts = []
    for i in range(11):
        x, y, a = inner_params[i]
        verts.append(hex_vertices(x, y, a))
        
    # Check overlaps
    for i in range(11):
        for j in range(i+1, 11):
            if hexes_overlap_penalty(verts[i], verts[j]) > 1e-6:
                return False
                
    # Check containment
    outer_rad = np.deg2rad(outer_angle)
    normals = np.array([
        [np.cos(outer_rad), np.sin(outer_rad)],
        [np.cos(outer_rad + np.pi/3), np.sin(outer_rad + np.pi/3)],
        [np.cos(outer_rad + 2*np.pi/3), np.sin(outer_rad + 2*np.pi/3)]
    ])
    r = np.sqrt(3) / 2 * L
    
    for v in verts:
        dots = np.abs(np.dot(normals, v.T))
        if np.max(dots) > r + 1e-6:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin

    # Initial guess: Valid lattice configuration
    centers = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, np.sqrt(3), 0.0],
        [-1.0, np.sqrt(3), 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -np.sqrt(3), 0.0],
        [1.0, -np.sqrt(3), 0.0],
        [4.0, 0.0, 0.0],
        [-4.0, 0.0, 0.0],
        [2.0, 2*np.sqrt(3), 0.0],
        [-2.0, -2*np.sqrt(3), 0.0]
    ]
    
    init_params = np.array(centers).flatten()
    init_params = np.append(init_params, 0.0) # outer angle
    
    # Calculate initial L for fallback
    init_verts = [hex_vertices(x, y, a) for x, y, a in centers]
    init_L = calculate_L(init_verts, 0.0)
    
    best_params = init_params.copy()
    best_score = objective(best_params)
    
    # Optimization
    try:
        # Use Nelder-Mead for better handling of non-smooth penalties than Powell sometimes
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        
        if time.time() < deadline:
             if res.fun < best_score:
                 best_params = res.x
                 best_score = res.fun
                 
    except Exception as e:
        pass # Keep initial if optimization fails

    # Extract results from best_params
    inner_params = best_params[:33].reshape(11, 3)
    outer_angle = best_params[33]
    
    # Calculate L for this configuration
    verts = [hex_vertices(x, y, a) for x, y, a in inner_params]
    L_final = calculate_L(verts, outer_angle)
    
    # Validate the optimized result
    if not is_valid(inner_params, outer_angle, L_final):
        # Fallback to initial valid configuration
        inner_params = np.array(centers).flatten().reshape(11, 3)
        outer_angle = 0.0
        L_final = init_L
        
    inner = []
    for i in range(11):
        x, y, a = inner_params[i]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), L_final, outer_angle
