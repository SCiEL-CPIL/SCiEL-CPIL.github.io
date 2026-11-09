import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Honeycomb packing using lattice (2i+j, j*sqrt3)
    # j=-1: i=-1,0,1 -> (-3,-sqrt3), (-1,-sqrt3), (1,-sqrt3)
    # j=0: i=-2,-1,0,1,2 -> (-4,0),(-2,0),(0,0),(2,0),(4,0)
    # j=1: i=-1,0,1 -> (-1,sqrt3),(1,sqrt3),(3,sqrt3)
    positions = []
    for j in [-1]:
        for i in [-1, 0, 1]:
            positions.append((2*i + j, j*sqrt3))
    for j in [0]:
        for i in [-2, -1, 0, 1, 2]:
            positions.append((2*i + j, j*sqrt3))
    for j in [1]:
        for i in [-1, 0, 1]:
            positions.append((2*i + j, j*sqrt3))
    
    assert len(positions) == N, f"Got {len(positions)} positions"
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (px, py) in positions:
        for k in range(6):
            angle = np.radians(60 * k)
            vx = px + np.cos(angle)
            vy = py + np.sin(angle)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    def enclosing_hex_size(params):
        cx, cy, theta = params
        max_val = 0.0
        for k in range(6):
            ang = np.radians(60 * k + 30) + theta
            cos_a = np.cos(ang)
            sin_a = np.sin(ang)
            vals = (all_vertices[:, 0] - cx) * cos_a + (all_vertices[:, 1] - cy) * sin_a
            mv = vals.max()
            if mv > max_val:
                max_val = mv
        return 2.0 * max_val / sqrt3
    
    # Optimize
    best_result = None
    best_size = float('inf')
    
    cx0, cy0 = all_vertices[:, 0].mean(), all_vertices[:, 1].mean()
    
    for theta0 in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
        if time.time() > deadline:
            break
        guess = [cx0, cy0, theta0]
        try:
            result = minimize(enclosing_hex_size, guess, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_size:
                best_size = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        best_result = [cx0, cy0, 0.0]
        best_size = enclosing_hex_size(best_result)
    
    cx, cy, theta = best_result
    side = best_size
    angle_deg = float(np.degrees(theta)) % 360.0
    
    inner = []
    for (px, py) in positions:
        inner.append([float(px), float(py), 0.0])
    
    return inner, (float(cx), float(cy)), float(side), angle_deg
# EVOLVE-BLOCK-END
