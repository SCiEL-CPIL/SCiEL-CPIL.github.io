import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate compact hexagonal cluster of 11 unit hexagons
    # Center + 6 first-ring + 4 second-ring
    centers = [(0.0, 0.0)]
    
    # First ring: 6 neighbors at distance sqrt3
    for i in range(6):
        angle = np.radians(60 * i)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    # Second ring: pick 4 from 12 positions at distance 2*sqrt3
    # Try different configurations
    second_ring_configs = [
        [0, 90, 180, 270],
        [30, 120, 210, 300],
        [0, 120, 240, 30],
        [60, 150, 240, 330],
    ]
    
    best_overall = None
    best_overall_val = float('inf')
    
    for config in second_ring_configs:
        if time.time() > deadline:
            break
        
        centers_try = [(0.0, 0.0)]
        for i in range(6):
            angle = np.radians(60 * i)
            x = sqrt3 * np.cos(angle)
            y = sqrt3 * np.sin(angle)
            centers_try.append((x, y))
        
        for angle_deg in config:
            angle = np.radians(angle_deg)
            x = 2 * sqrt3 * np.cos(angle)
            y = 2 * sqrt3 * np.sin(angle)
            centers_try.append((x, y))
        
        centers_arr = np.array(centers_try)
        
        def compute_min_side(cx, cy, theta, centers_arr=centers_arr):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            max_proj = 0.0
            
            for ix, iy in centers_arr:
                for k in range(6):
                    angle = np.radians(60 * k)
                    vx = ix + np.cos(angle)
                    vy = iy + np.sin(angle)
                    dx = vx - cx
                    dy = vy - cy
                    lx = dx * cos_t + dy * sin_t
                    ly = -dx * sin_t + dy * cos_t
                    for m in range(6):
                        angle_m = np.radians(30 + 60 * m)
                        nx = np.cos(angle_m)
                        ny = np.sin(angle_m)
                        proj = lx * nx + ly * ny
                        if proj > max_proj:
                            max_proj = proj
                        if -proj > max_proj:
                            max_proj = -proj
            return max_proj * 2.0 / sqrt3
        
        def objective(params, centers_arr=centers_arr):
            cx, cy, theta = params
            return compute_min_side(cx, cy, theta, centers_arr)
        
        # Optimize outer hexagon
        best_val = float('inf')
        best_params = None
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                x0_try = [0.0, 0.0, 0.0]
            else:
                x0_try = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0, 2*np.pi)]
            
            res = minimize(objective, x0_try, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        
        if best_params is not None and best_val < best_overall_val:
            best_overall_val = best_val
            best_overall = (centers_try, best_params)
    
    if best_overall is None:
        # Fallback
        centers_final = [(0.0, 0.0)]
        for i in range(6):
            angle = np.radians(60 * i)
            centers_final.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
        for angle_deg in [0, 90, 180, 270]:
            angle = np.radians(angle_deg)
            centers_final.append((2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)))
        cx, cy, theta = 0.0, 0.0, 0.0
        side = 10.0  # rough estimate
    else:
        centers_final, params = best_overall
        cx, cy, theta = params
        side = best_overall_val
    
    inner = [[c[0], c[1], 0.0] for c in centers_final]
    return inner, (cx, cy), side, np.degrees(theta)
# EVOLVE-BLOCK-END
