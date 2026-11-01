import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: hexagonal lattice
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # 6 around center at distance 2
    for k in range(6):
        angle = np.radians(60 * k)
        positions.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    # 4 more at distance 4
    for k in [0, 2, 4, 5]:
        angle = np.radians(60 * k)
        positions.append((4.0 * np.cos(angle), 4.0 * np.sin(angle)))
    
    px = np.array([p[0] for p in positions])
    py = np.array([p[1] for p in positions])
    angles = np.zeros(N)
    
    def outer_hexagon_side(px_arr, py_arr, cx, cy, theta_deg):
        theta = np.radians(theta_deg)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        max_proj = 0.0
        for i in range(len(px_arr)):
            dx = px_arr[i] - cx
            dy = py_arr[i] - cy
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            for k in range(6):
                angle = np.radians(30 + 60 * k)
                nx = np.cos(angle)
                ny = np.sin(angle)
                proj = abs(rx * nx + ry * ny)
                if proj > max_proj:
                    max_proj = proj
        
        apothem = max_proj
        s = apothem * 2 / np.sqrt(3)
        return s
    
    def objective(params):
        px_new = params[0:N]
        py_new = params[N:2*N]
        angles_new = params[2*N:3*N]
        cx = params[3*N]
        cy = params[3*N + 1]
        theta_outer = params[3*N + 2]
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((px_new[i] - px_new[j])**2 + (py_new[i] - py_new[j])**2)
                if d < 2.0:
                    overlap_penalty += (2.0 - d) ** 2
        
        s = outer_hexagon_side(px_new, py_new, cx, cy, theta_outer)
        
        return s + 1000.0 * overlap_penalty
    
    x0 = np.concatenate([px, py, angles, [0.0, 0.0, 0.0]])
    
    best_result = None
    best_obj = float('inf')
    
    np.random.seed(42)
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params_init = x0.copy()
        else:
            perturbation = np.random.normal(0, 0.5, size=2*N)
            params_init = x0.copy()
            params_init[0:N] += perturbation[0:N]
            params_init[N:2*N] += perturbation[N:2*N]
            params_init[2*N:3*N] = np.random.uniform(-30, 30, size=N)
            params_init[3*N] = np.random.uniform(-1, 1)
            params_init[3*N + 1] = np.random.uniform(-1, 1)
            params_init[3*N + 2] = np.random.uniform(-30, 30)
        
        try:
            result = minimize(objective, params_init, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        best_result = x0
    
    px_final = best_result[0:N]
    py_final = best_result[N:2*N]
    angles_final = best_result[2*N:3*N]
    cx_final = best_result[3*N]
    cy_final = best_result[3*N + 1]
    theta_final = best_result[3*N + 2]
    
    s_final = outer_hexagon_side(px_final, py_final, cx_final, cy_final, theta_final)
    
    inner = []
    for i in range(N):
        inner.append([float(px_final[i]), float(py_final[i]), float(angles_final[i])])
    
    return inner, (float(cx_final), float(cy_final)), float(s_final), float(theta_final)
# EVOLVE-BLOCK-END
