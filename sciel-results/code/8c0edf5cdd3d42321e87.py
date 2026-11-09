"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def _min_outer_side(positions, angles, n_inner):
    centers = np.array(positions)
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    
    all_points = []
    for i in range(n_inner):
        verts = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [0.0, 1.0],
        [-np.cos(np.pi/6), np.sin(np.pi/6)],
        [-np.cos(np.pi/6), -np.sin(np.pi/6)],
        [0.0, -1.0],
        [np.cos(np.pi/6), -np.sin(np.pi/6)]
    ])
    
    rel = all_points - np.array([cx, cy])
    max_dots = np.zeros(6)
    min_dots = np.zeros(6)
    for j in range(6):
        dots = rel @ normals[j]
        max_dots[j] = np.max(dots)
        min_dots[j] = np.min(dots)
    
    required = np.maximum(max_dots, -min_dots)
    S = np.max(required) / (np.sqrt(3) / 2)
    return S, cx, cy

def _objective(params):
    n = N
    positions = np.zeros((n, 2))
    angles = np.zeros(n)
    for i in range(n):
        positions[i, 0] = params[2*i]
        positions[i, 1] = params[2*i+1]
        angles[i] = params[2*n + i]
    S, cx, cy = _min_outer_side(positions, angles, n)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    sqrt3 = np.sqrt(3)
    
    # Initial: hexagonal lattice 3-4-3-1
    positions = []
    angles = []
    for k in range(-1, 2):
        positions.append([k * sqrt3, 0.0])
        angles.append(0.0)
    for k in range(-1, 3):
        positions.append([k * sqrt3 + sqrt3/2, 1.5])
        angles.append(0.0)
    for k in range(-1, 2):
        positions.append([k * sqrt3, 3.0])
        angles.append(0.0)
    positions.append([0.0, -1.5])
    angles.append(0.0)
    
    n = len(positions)
    params = np.zeros(3*n)
    for i in range(n):
        params[2*i] = positions[i][0]
        params[2*i+1] = positions[i][1]
        params[2*n + i] = angles[i]
    
    best_S = _objective(params)
    best_params = params.copy()
    
    if time.time() < deadline - 20:
        try:
            result = minimize(_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 300, 'maxfev': 1500})
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'maxfev': 1000})
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x
        except Exception:
            pass
    
    n = N
    pos = np.zeros((n, 2))
    ang = np.zeros(n)
    for i in range(n):
        pos[i, 0] = best_params[2*i]
        pos[i, 1] = best_params[2*i+1]
        ang[i] = best_params[2*n + i]
    
    S, cx, cy = _min_outer_side(pos, ang, n)
    
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(n)]
    return inner, (cx, cy), S, 0.0
# EVOLVE-BLOCK-END
