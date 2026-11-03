import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0
    
    def axial_to_pixel(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    hex_angles = np.array([np.pi / 3 * i for i in range(6)])
    hex_vertex_offsets = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def compute_all_vertices(centers):
        expanded = np.repeat(centers, 6, axis=0)
        return expanded + hex_vertex_offsets
    
    def min_enclosing_side(all_verts, theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        sqrt3_2 = np.sqrt(3) / 2
        
        dirs = np.array([
            [cos_t, sin_t],
            [-0.5*cos_t - sqrt3_2*sin_t, -0.5*sin_t + sqrt3_2*cos_t],
            [-0.5*cos_t + sqrt3_2*sin_t, -0.5*sin_t - sqrt3_2*cos_t],
            [-cos_t, -sin_t],
            [0.5*cos_t + sqrt3_2*sin_t, 0.5*sin_t - sqrt3_2*cos_t],
            [0.5*cos_t - sqrt3_2*sin_t, 0.5*sin_t + sqrt3_2*cos_t]
        ])
        
        projections = all_verts @ dirs.T
        ranges = np.max(projections, axis=0) - np.min(projections, axis=0)
        max_range = np.max(ranges)
        return max_range / np.sqrt(3)
    
    def optimize_for_centers(centers):
        all_verts = compute_all_vertices(centers)
        
        def cost(theta):
            return min_enclosing_side(all_verts, theta)
        
        result = minimize_scalar(cost, bounds=(0, 2*np.pi), method='bounded',
                                options={'xatol': 1e-6, 'maxiter': 200})
        
        theta_opt = result.x
        S_opt = result.fun
        
        cos_t = np.cos(theta_opt)
        sin_t = np.sin(theta_opt)
        sqrt3_2 = np.sqrt(3) / 2
        dirs = np.array([
            [cos_t, sin_t],
            [-0.5*cos_t - sqrt3_2*sin_t, -0.5*sin_t + sqrt3_2*cos_t],
            [-0.5*cos_t + sqrt3_2*sin_t, -0.5*sin_t - sqrt3_2*cos_t],
            [-cos_t, -sin_t],
            [0.5*cos_t + sqrt3_2*sin_t, 0.5*sin_t - sqrt3_2*cos_t],
            [0.5*cos_t - sqrt3_2*sin_t, 0.5*sin_t + sqrt3_2*cos_t]
        ])
        
        projections = all_verts @ dirs.T
        midpoints = (np.max(projections, axis=0) + np.min(projections, axis=0)) / 2
        c, _, _, _ = np.linalg.lstsq(dirs.T, midpoints, rcond=None)
        
        return S_opt, c[0], c[1], theta_opt
    
    best_S = float('inf')
    best_config = None
    
    hex19_coords = [(q, r) for q in range(-2, 3) for r in range(-2, 3) if abs(q + r) <= 2]
    hex19_pixels = [axial_to_pixel(q, r) for q, r in hex19_coords]
    
    for combo in combinations(range(19), 11):
        if time.time() > deadline - 25:
            break
        
        centers = np.array([hex19_pixels[i] for i in combo])
        S, cx, cy, theta = optimize_for_centers(centers)
        
        if S < best_S:
            best_S = S
            inner = []
            for i in combo:
                pos = hex19_pixels[i]
                inner.append([float(pos[0]), float(pos[1]), 0.0])
            best_config = (inner, (float(cx), float(cy)), float(S), float(np.degrees(theta) % 360))
    
    if best_config is None:
        inner = [[np.sqrt(3) * k, 0.0, 0.0] for k in range(11)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_config
# EVOLVE-BLOCK-END
