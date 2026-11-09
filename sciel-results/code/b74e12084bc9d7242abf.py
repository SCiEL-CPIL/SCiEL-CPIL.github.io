import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_unit_hex_vertices():
    # Vertices of a unit regular hexagon centered at origin
    # Side length 1. Distance from center to vertex is 1.
    angles = np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: Honeycomb
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    # 7 central hexagons
    centers = [
        np.array([0.0, 0.0]),
        v1,
        v2,
        v2 - v1,
        -v1,
        -v2,
        v1 - v2
    ]
    
    # 5 from the next layer (Layer 2)
    # Pick 5 that are spread out
    layer2_candidates = [
        2*v1,
        2*v2,
        2*(v2-v1),
        v1+v2,
        2*v2-v1,
        -v1+v2,
        -2*v1,
        -2*v2,
        -2*(v2-v1),
        -v1-v2,
        -2*v2+v1,
        v1-(v2-v1) # = 2v1-v2
    ]
    # Select 5
    # Indices 0, 1, 2, 3, 4
    centers.extend(layer2_candidates[:5])
    
    # Flatten to array for optimization
    # x, y, angle for each
    x0 = []
    for c in centers:
        x0.extend([c[0], c[1], 0.0])
    x0 = np.array(x0)
    
    unit_verts = get_unit_hex_vertices()
    
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10
            
        S_max = 0.0
        # Outer hexagon normals
        # Assume outer hexagon is fixed at angle 0
        # Normals at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            theta = params[3*i+2]
            
            # Rotate unit vertices
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            verts = unit_verts @ rot.T + np.array([cx, cy])
            
            # Project onto normals
            # dots = verts @ normals.T
            # max_abs = np.max(np.abs(dots))
            # S_max = max(S_max, max_abs)
            
            # Vectorized for speed
            dots = np.dot(verts, normals.T)
            max_abs = np.max(np.abs(dots))
            if max_abs > S_max:
                S_max = max_abs
                
        # S = 2/sqrt(3) * S_max
        return (2.0 / sqrt3) * S_max

    # Try to optimize
    best_x = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})
        if time.time() < deadline:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass
        
    # Try a few random restarts if time permits
    num_restarts = 5
    for _ in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        # Perturb initial solution
        x_rand = x0 + np.random.normal(0, 0.5, size=x0.shape)
        try:
            res = minimize(objective, x_rand, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
            if time.time() < deadline:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            pass

    # Extract results
    inner = []
    for i in range(N):
        x = best_x[3*i]
        y = best_x[3*i+1]
        angle = np.rad2deg(best_x[3*i+2])
        inner.append([x, y, angle])
        
    center = [0.0, 0.0]
    side = best_val
    angle_outer = 0.0
    
    return inner, center, side, angle_outer

