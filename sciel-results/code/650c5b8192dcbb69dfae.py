"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Precompute hexagon vertex offsets for a unit hexagon (side=1, circumradius=1)
    hex_angles = np.radians([0, 60, 120, 180, 240, 300])
    hex_offsets = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def compute_all_vertices(r1, r2):
        vertices = []
        # Central hex
        vertices.append(hex_offsets)
        # 5 inner ring
        for j in range(5):
            angle = np.radians(72 * j)
            cx, cy = r1 * np.cos(angle), r1 * np.sin(angle)
            vertices.append(hex_offsets + np.array([cx, cy]))
        # 6 outer ring
        for j in range(6):
            angle = np.radians(60 * j)
            cx, cy = r2 * np.cos(angle), r2 * np.sin(angle)
            vertices.append(hex_offsets + np.array([cx, cy]))
        return np.vstack(vertices)
    
    def objective(params):
        r1, r2, theta = params
        if r1 < 0 or r2 < 0:
            return 1e10
        if r2 < r1:
            return 1e10
        vertices = compute_all_vertices(r1, r2)
        theta_rad = np.radians(theta)
        normal_angles = theta_rad + np.radians(30) + np.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        dots = vertices @ normals.T
        max_dots = np.max(dots, axis=1)
        max_dot = np.max(max_dots)
        return 2.0 * max_dot / np.sqrt(3.0)
    
    # Try multiple initial points
    best_S = 1e10
    best_params = None
    initial_guesses = [
        [2.0, 4.0, 0.0],
        [1.8, 3.5, 15.0],
        [2.2, 4.5, 30.0],
        [1.5, 3.0, 0.0],
        [2.5, 5.0, 0.0],
    ]
    
    for x0 in initial_guesses:
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x
        except:
            pass
    
    if best_params is None:
        best_params = np.array([2.0, 4.0, 0.0])
        best_S = objective(best_params)
    
    r1, r2, theta = best_params
    if r1 < 0: r1 = 0
    if r2 < 0: r2 = 0
    if r2 < r1: r2 = r1
    
    inner = []
    inner.append([0.0, 0.0, 0.0])
    for j in range(5):
        angle = np.radians(72 * j)
        cx, cy = r1 * np.cos(angle), r1 * np.sin(angle)
        inner.append([float(cx), float(cy), 0.0])
    for j in range(6):
        angle = np.radians(60 * j)
        cx, cy = r2 * np.cos(angle), r2 * np.sin(angle)
        inner.append([float(cx), float(cy), 0.0])
    
    S = float(best_S)
    
    return inner, (0.0, 0.0), S, float(theta)
# EVOLVE-BLOCK-END
