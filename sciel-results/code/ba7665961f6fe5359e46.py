import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600 # seconds
    
    # Initial guess: 7+4 cluster
    c0 = [0, 0]
    c1 = [np.sqrt(3), 0]
    c2 = [np.sqrt(3)/2, 1.5]
    c3 = [-np.sqrt(3)/2, 1.5]
    c4 = [-np.sqrt(3), 0]
    c5 = [-np.sqrt(3)/2, -1.5]
    c6 = [np.sqrt(3)/2, -1.5]
    
    d2 = 2 * np.sqrt(3)
    c7 = [d2 * np.cos(np.pi/6), d2 * np.sin(np.pi/6)]
    c8 = [d2 * np.cos(5*np.pi/6), d2 * np.sin(5*np.pi/6)]
    c9 = [d2 * np.cos(7*np.pi/6), d2 * np.sin(7*np.pi/6)]
    c10 = [d2 * np.cos(11*np.pi/6), d2 * np.sin(11*np.pi/6)]
    
    centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    def get_vertices(centers):
        verts = np.zeros((11, 6, 2))
        for i in range(11):
            cx, cy = centers[i]
            for j in range(6):
                angle = np.pi / 3 * j
                verts[i, j, 0] = cx + np.cos(angle)
                verts[i, j, 1] = cy + np.sin(angle)
        return verts

    def calc_S(centers):
        verts = get_vertices(centers)
        pts = verts.reshape(-1, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]
        S1 = (2 / np.sqrt(3)) * np.abs(ys)
        S2 = np.abs(xs) + (1 / np.sqrt(3)) * np.abs(ys)
        S = np.maximum(S1, S2)
        return np.max(S)

    def objective(x):
        centers = x.reshape(11, 2)
        S = calc_S(centers)
        penalty = 0
        min_dist = np.sqrt(3)
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    penalty += (min_dist - d)**2
        return S + 100 * penalty

    x0 = centers.flatten()
    
    # Run optimization
    # Nelder-Mead is good for small N
    best_S = calc_S(centers)
    best_centers = centers
    
    try:
        # Use multiple restarts if time permits? 
        # For now, single run with high maxiter
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
        
        if result.fun < best_S:
             best_centers = result.x.reshape(11, 2)
             best_S = calc_S(best_centers)
             
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 10:
         pass # Return best found
        
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0
        
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, float(best_S)

# EVOLVE-BLOCK-END
