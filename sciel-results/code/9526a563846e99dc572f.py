import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 150.0
SQR3 = np.sqrt(3)

def construct():
    # Variables: x1, y1, ..., xN, yN, L
    # Fixed: Outer center (0,0), Outer angle 0, Inner angles 0
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            # Return a large value to stop optimization if possible, 
            # but minimize doesn't handle exceptions well in objective.
            # We rely on the outer loop to check time.
            pass
        
        positions = params[:2*N].reshape(N, 2)
        L = params[2*N]
        
        if L <= 0:
            return 1e6
        
        # 1. Penalty for inner hexagons being outside outer hexagon
        # Outer hexagon: center (0,0), side L, rot 0.
        # Vertices of inner hex i (center cx, cy, side 1, rot 0):
        # We check if any vertex of inner hex is outside outer hex.
        
        # Pre-compute unit hex vertices (side 1, rot 0)
        # Angles: 0, 60, 120, 180, 240, 300
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
        
        penalty_out = 0.0
        for i in range(N):
            cx, cy = positions[i]
            # Translate unit vertices to cx, cy
            verts = unit_verts + np.array([cx, cy])
            
            for vx, vy in verts:
                # Check containment in outer hex (side L, rot 0)
                # Conditions:
                # |y| <= sqrt(3)/2 * L
                # |sqrt(3)x + y| <= sqrt(3) * L
                # |sqrt(3)x - y| <= sqrt(3) * L
                
                v1 = abs(vy) - 0.5 * SQR3 * L
                v2 = abs(SQR3 * vx + vy) - SQR3 * L
                v3 = abs(SQR3 * vx - vy) - SQR3 * L
                
                if v1 > 0: penalty_out += v1**2
                if v2 > 0: penalty_out += v2**2
                if v3 > 0: penalty_out += v3**2
                
        # 2. Penalty for inner hexagons overlapping
        # Min distance between centers for non-overlap (rot 0) is sqrt(3)
        penalty_in = 0.0
        min_dist = SQR3
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty_in += (min_dist - d)**2
                    
        return L + 100.0 * penalty_out + 100.0 * penalty_in

    def run_optimization(initial_pos, L_init):
        x0 = np.concatenate((initial_pos.flatten(), [L_init]))
        
        # Bounds: L > 1, positions can be anything but let's bound them to [-10, 10]
        bounds = [(-10, 10)] * (2*N) + [(1.0, 10.0)]
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-6})
        
        if res.success:
            return res.x
        else:
            return x0 # Return initial if failed

    # Generate initial hex grid points
    # Basis vectors for hex grid with spacing sqrt(3)
    u = np.array([SQR3, 0])
    v = np.array([SQR3/2, 1.5])
    
    points = []
    for k in range(-4, 5):
        for m in range(-4, 5):
            p = k * u + m * v
            points.append(p)
            
    # Sort by distance to origin
    points.sort(key=lambda p: np.dot(p, p))
    
    # Take first N points
    initial_pos = np.array(points[:N])
    
    # Initial L estimate: max distance from origin to any point + 1 (for hex radius)
    max_r = np.max(np.linalg.norm(initial_pos, axis=1))
    L_init = max_r + 1.0
    
    best_params = None
    best_val = float('inf')
    
    # Run optimization
    if time.time() - START_TIME < TIME_LIMIT - 10:
        res_x = run_optimization(initial_pos, L_init)
        val = objective(res_x)
        if val < best_val:
            best_val = val
            best_params = res_x
            
    # Try a few random perturbations if time allows
    start_restart = time.time()
    num_restarts = 0
    while time.time() - start_restart < 30 and (time.time() - START_TIME) < TIME_LIMIT - 10:
        # Perturb initial positions
        noise = np.random.normal(0, 0.5, (N, 2))
        perturbed_pos = initial_pos + noise
        L_init_r = L_init + np.random.uniform(0, 1)
        
        res_x = run_optimization(perturbed_pos, L_init_r)
        val = objective(res_x)
        
        if val < best_val:
            best_val = val
            best_params = res_x
            
        num_restarts += 1
        if num_restarts > 10: break # Cap restarts

    if best_params is None:
        # Fallback
        inner = [[initial_pos[i][0], initial_pos[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), L_init, 0.0

    # Extract results
    positions = best_params[:2*N].reshape(N, 2)
    L_final = best_params[2*N]
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), 0.0])
        
    return inner, (0.0, 0.0), float(L_final), 0.0
