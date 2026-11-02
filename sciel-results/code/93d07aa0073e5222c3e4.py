import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initial configuration: a 3-4-4 pattern in axial coordinates
    # This gives a roughly hexagonal cluster
    axial_init = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    init_pts = axial_to_cartesian(axial_init)
    
    # Center the initial configuration
    init_center = np.mean(init_pts, axis=0)
    init_pts = init_pts - init_center
    
    # Variables: [S, cx, cy, x_0, y_0, x_1, y_1, ..., x_10, y_10]
    # Total: 3 + 2*N = 25 variables
    n_vars = 3 + 2 * N
    
    def objective(vars):
        if not _time_ok():
            return 1e10
        S = vars[0]
        cx = vars[1]
        cy = vars[2]
        pts = vars[3:].reshape(N, 2)
        
        # Outer hexagon constraint: for each point and each of 6 directions,
        # |n_k . (p_i - c)| <= S * APOTHEM
        # We use a soft violation penalty
        penalty = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            for i in range(N):
                dx = pts[i, 0] - cx
                dy = pts[i, 1] - cy
                proj = dx * nx + dy * ny
                violation = max(0, abs(proj) - S * APOTHEM)
                penalty += violation ** 2
        
        # Non-overlap constraint: ||p_i - p_j|| >= sqrt(3)
        min_dist = math.sqrt(3)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                violation = max(0, min_dist - dist)
                penalty += violation ** 2
        
        return S + 10.0 * penalty
    
    def objective_gradient(vars):
        # We'll use numerical gradient for simplicity, or just use SLSQP with finite differences
        return None
    
    x0 = np.zeros(n_vars)
    x0[0] = 5.0  # Initial S guess
    x0[1] = 0.0  # cx
    x0[2] = 0.0  # cy
    for i in range(N):
        x0[3 + 2*i] = init_pts[i, 0]
        x0[3 + 2*i + 1] = init_pts[i, 1]
    
    bounds = [(0.1, 20.0), (-10, 10), (-10, 10)] + [(-10, 10)] * (2 * N)
    
    best_result = None
    best_S = float('inf')
    
    # Try multiple initial configurations
    initial_configs = [axial_init]
    
    # Add a few more initial configs
    axial_init2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    initial_configs.append(axial_init2)
    
    axial_init3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    initial_configs.append(axial_init3)
    
    axial_init4 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    initial_configs.append(axial_init4)
    
    for config in initial_configs:
        if not _time_ok():
            break
        
        init_pts = axial_to_cartesian(config)
        init_center = np.mean(init_pts, axis=0)
        init_pts = init_pts - init_center
        
        x0 = np.zeros(n_vars)
        x0[0] = 5.0
        x0[1] = 0.0
        x0[2] = 0.0
        for i in range(N):
            x0[3 + 2*i] = init_pts[i, 0]
            x0[3 + 2*i + 1] = init_pts[i, 1]
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < best_S:
                S_val = result.x[0]
                cx_val = result.x[1]
                cy_val = result.x[2]
                pts_val = result.x[3:].reshape(N, 2)
                
                # Verify the solution
                max_proj = 0
                for k in range(6):
                    ang = math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    for i in range(N):
                        dx = pts_val[i, 0] - cx_val
                        dy = pts_val[i, 1] - cy_val
                        proj = abs(dx * nx + dy * ny)
                        max_proj = max(max_proj, proj)
                
                required_S = max_proj / APOTHEM
                
                # Check non-overlap
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pts_val[i, 0] - pts_val[j, 0]
                        dy = pts_val[i, 1] - pts_val[j, 1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < math.sqrt(3) - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and required_S < best_S:
                    best_S = required_S
                    best_result = (S_val, cx_val, cy_val, pts_val)
                    
        except Exception as e:
            pass
    
    # If optimization failed, use a fallback
    if best_result is None:
        # Use the initial 3-4-4 configuration
        init_pts = axial_to_cartesian(axial_init)
        init_center = np.mean(init_pts, axis=0)
        cx = init_center[0]
        cy = init_center[1]
        
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            for x, y in init_pts:
                dx = x - cx
                dy = y - cy
                proj = abs(dx * nx + dy * ny)
                max_proj = max(max_proj, proj)
        
        best_S = max_proj / APOTHEM
        best_result = (best_S, cx, cy, init_pts)
    
    S_val, cx_val, cy_val, pts_val = best_result
    
    # Use a small safety margin
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for i in range(N):
        sx = cx_val + (pts_val[i, 0] - cx_val) * SCALE
        sy = cy_val + (pts_val[i, 1] - cy_val) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        for i in range(N):
            dx = pts_scaled[i, 0] - cx_val
            dy = pts_scaled[i, 1] - cy_val
            proj = abs(dx * nx + dy * ny)
            max_proj = max(max_proj, proj)
    
    S_final = max_proj / APOTHEM
    
    inner = []
    for i in range(N):
        inner.append([float(pts_scaled[i, 0]), float(pts_scaled[i, 1]), 0.0])
    
    center = [float(cx_val), float(cy_val)]
    side = float(S_final)
    rotation = 0.0
    
    return inner, center, side, rotation
