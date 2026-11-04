import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    sqrt3 = np.sqrt(3.0)
    
    # Compact 11-hexagon cluster in hexagonal lattice
    axial_positions = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0), (0,  0), (1,  0),
        (-1,  1), (0,  1), (1,  1),
        ( 0,  2), (1,  2)
    ]
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    points = np.array([axial_to_cartesian(q, r) for q, r in axial_positions])
    points_centered = points - points.mean(axis=0)
    
    # Edge normals of outer hexagon (vertices at 0, 60, 120, ...)
    edge_normal_angles = np.array([(30 + k * 60) * np.pi / 180.0 for k in range(6)])
    n_k = np.column_stack([np.cos(edge_normal_angles), np.sin(edge_normal_angles)])
    m_angles = np.arange(6) * np.pi / 3.0
    
    def outer_hex_side_length(theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_points = points_centered @ R.T
        
        phi_values = edge_normal_angles - theta
        cos_vals = np.cos(phi_values[:, np.newaxis] - m_angles[np.newaxis, :])
        h_inner = np.max(cos_vals, axis=1)
        
        dots = rotated_points @ n_k.T
        max_val = np.max(dots + h_inner[np.newaxis, :])
        s = max_val / (sqrt3 / 2.0)
        return s
    
    best_s = float('inf')
    best_theta = 0.0
    
    theta_grid = np.linspace(0, np.pi / 3, 61)
    for theta in theta_grid:
        s = outer_hex_side_length(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    if time.time() < deadline:
        lo = max(0.0, best_theta - np.pi / 60)
        hi = min(np.pi / 3, best_theta + np.pi / 60)
        result = minimize_scalar(outer_hex_side_length, bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
        if result.fun < best_s:
            best_s = result.fun
            best_theta = result.x
    
    cos_t, sin_t = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    final_points = points_centered @ R.T
    
    inner = [[float(p[0]), float(p[1]), float(np.degrees(best_theta))] for p in final_points]
    
    return inner, (0.0, 0.0), float(best_s), 0.0
# EVOLVE-BLOCK-END
