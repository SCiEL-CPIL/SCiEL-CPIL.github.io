import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    
    # Define the centers of the 11 inner hexagons
    centers = []
    centers.append((0.0, 0.0))
    
    d1 = np.sqrt(3)
    for i in range(6):
        angle = np.pi / 3 * i
        x = d1 * np.cos(angle)
        y = d1 * np.sin(angle)
        centers.append((x, y))
        
    d2 = 2 * np.sqrt(3)
    for i in [0, 1, 2, 3]:
        angle = np.pi / 3 * i
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        centers.append((x, y))
        
    vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.pi / 3 * k
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            vertices.append((vx, vy))
            
    vertices = np.array(vertices)
    centroid = np.mean(vertices, axis=0)
    
    best_S = float('inf')
    best_theta = 0.0
    best_center = tuple(centroid)
    
    # Search theta
    # We can use a finer grid or optimization.
    # 0.1 deg is 600 steps. Very fast.
    for theta_deg in np.arange(0, 60, 0.1):
        if time.time() - start_time > 140:
            break
        theta = np.radians(theta_deg)
        
        normals = []
        for k in range(6):
            angle = theta + np.pi / 6 + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
            
        max_A = 0.0
        for n in normals:
            n_vec = np.array(n)
            proj = vertices @ n_vec
            proj_centered = proj - centroid @ n_vec
            max_abs = np.max(np.abs(proj_centered))
            if max_abs > max_A:
                max_A = max_abs
                
        S = 2 * max_A / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_theta = theta
            best_center = tuple(centroid)
            
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, best_center, best_S, np.degrees(best_theta)
