import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin

# Vertices of a unit regular hexagon, centered at (0,0), angle 0
# Side length 1.
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

# Normal vectors for the outer hexagon (angle 0)
# Faces are perpendicular to these normals.
OUTER_NORMALS = np.array([
    [np.cos(np.pi/6), np.sin(np.pi/6)],
    [np.cos(np.pi/2), np.sin(np.pi/2)],
    [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
    [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
    [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
    [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
])

def calculate_S_and_penalty(params):
    """
    params: array of length 2*N, [x0, y0, x1, y1, ..., x10, y10]
    Returns: (S, penalty)
    """
    centers = params.reshape(N, 2)
    
    # Calculate S (outer side length)
    # For each of the 6 directions, find the max projection of any vertex of any inner hexagon
    max_projs = np.zeros(6)
    for i in range(N):
        c = centers[i]
        verts = c + HEX_VERTS
        # Project vertices onto the 6 normals
        projs = verts @ OUTER_NORMALS.T
        max_projs = np.maximum(max_projs, projs.max(axis=0))
    
    # S * sqrt(3) / 2 >= max_proj_val
    max_proj_val = np.max(max_projs)
    S = (2.0 / np.sqrt(3.0)) * max_proj_val
    
    # Calculate Overlap Penalty
    # Min distance between centers for non-overlapping aligned unit hexagons is sqrt(3)
    min_dist = np.sqrt(3.0)
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2
                
    return S, penalty

def objective(params):
    S, penalty = calculate_S_and_penalty(params)
    return S + 100.0 * penalty

def construct():
    # Initial guess: Hexagonal lattice
    sqrt3 = np.sqrt(3.0)
    init_centers = []
    init_centers.append([0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3.0
        init_centers.append([sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)])
    
    # Add 4 more
    for k in range(4):
        angle = (k * 2 + 1) * np.pi / 6.0 # 30, 90, 150, 210
        init_centers.append([2.0 * sqrt3 * np.cos(angle), 2.0 * sqrt3 * np.sin(angle)])
        
    init_params = np.array(init_centers).flatten()
    
    best_S = np.inf
    best_params = init_params
    
    # Run optimization
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Perturb initial guess
        noise = 0.2 * np.random.randn(len(init_params))
        params = init_params + noise
        
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        S, penalty = calculate_S_and_penalty(res.x)
        
        # Check if valid (penalty is very small)
        if penalty < 1e-4:
            if S < best_S:
                best_S = S
                best_params = res.x
                
    # Final result
    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        inner.append([x, y, 0.0])
        
    # If best_S is inf, something went wrong. Return a valid fallback.
    if np.isinf(best_S):
        inner = []
        for i in range(N):
            x = init_params[2*i]
            y = init_params[2*i+1]
            inner.append([x, y, 0.0])
        S, _ = calculate_S_and_penalty(init_params)
        best_S = S

    return inner, [0.0, 0.0], best_S, 0.0
