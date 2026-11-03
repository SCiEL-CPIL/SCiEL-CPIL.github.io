import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_v = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    y_v = np.array([0.0, np.sqrt(3)/2, np.sqrt(3)/2, 0.0, -np.sqrt(3)/2, -np.sqrt(3)/2])
    x_rot = x_v * cos_a - y_v * sin_a
    y_rot = x_v * sin_a + y_v * cos_a
    return np.column_stack((x_rot + cx, y_rot + cy))

def compute_min_side(params):
    inner_pos = params[:N*2].reshape(N, 2)
    inner_angles = params[N*2:N*3]
    outer_angle = params[N*3]
    
    # Overlap penalty
    penalty = 0.0
    min_dist = np.sqrt(3)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_pos[i] - inner_pos[j])
            if d < min_dist:
                penalty += (min_dist - d)**2
                
    if penalty > 1.0:
        return 1000.0 + penalty # Early exit for large overlaps

    all_verts = []
    for i in range(N):
        v = get_hex_vertices(inner_pos[i][0], inner_pos[i][1], inner_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    base_angle = np.deg2rad(outer_angle)
    normals = np.array([np.array([np.cos(base_angle + k*np.pi/3), np.sin(base_angle + k*np.pi/3)]) for k in range(6)])
    
    projections = all_verts @ normals.T
    M = np.max(projections, axis=0)
    
    def objective_c(c):
        return np.max(M - normals @ c)
        
    c0 = np.mean(all_verts, axis=0)
    res_c = minimize(objective_c, c0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    min_val = res_c.fun
    side = 2 * min_val / np.sqrt(3)
    
    return side + 10.0 * penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initialization
    pos = []
    angles = []
    # 3 rows
    # Row 0: 4 hex
    # Row 1: 4 hex
    # Row 2: 3 hex
    for r in range(3):
        if r < 2:
            xs = np.linspace(-1.5*np.sqrt(3), 1.5*np.sqrt(3), 4)
        else:
            xs = np.linspace(-np.sqrt(3), np.sqrt(3), 3)
        for x in xs:
            y = r * 1.5
            pos.append([x, y])
            angles.append(0.0)
            
    pos = np.array(pos)
    angles = np.array(angles)
    x0 = np.concatenate([pos.flatten(), angles, [0.0]])
    
    def objective(params):
        return compute_min_side(params)
        
    best_res = None
    best_score = np.inf
    
    # Powell optimization
    # Use time check
    for restart in range(3):
        if time.time() > deadline - 30.0:
            break
            
        if restart == 0:
            current_x = x0
        else:
            current_x = x0 + np.random.normal(0, 0.5, x0.size)
            
        # Use Nelder-Mead for the outer loop? Powell is better for 34 vars.
        # But Powell can be slow.
        # Let's try Nelder-Mead with tight settings or Powell with maxiter limit.
        res = minimize(objective, current_x, method='Powell', options={'maxiter': 1000, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_res = res
            
        if time.time() > deadline - 10.0:
            break
            
    if best_res is None:
        return [[0,0,0] for _ in range(N)], [0,0], 10.0, 0.0
        
    x_opt = best_res.x
    inner_pos = x_opt[:N*2].reshape(N, 2)
    inner_angles = x_opt[N*2:N*3]
    outer_angle = x_opt[N*3]
    
    # Final polish: Re-compute center and side accurately
    all_verts = []
    for i in range(N):
        v = get_hex_vertices(inner_pos[i][0], inner_pos[i][1], inner_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    base_angle = np.deg2rad(outer_angle)
    normals = np.array([np.array([np.cos(base_angle + k*np.pi/3), np.sin(base_angle + k*np.pi/3)]) for k in range(6)])
    
    projections = all_verts @ normals.T
    M = np.max(projections, axis=0)
    
    def objective_c(c):
        return np.max(M - normals @ c)
        
    c0 = np.mean(all_verts, axis=0)
    res_c = minimize(objective_c, c0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    center = res_c.x
    side = 2 * res_c.fun / np.sqrt(3)
    
    inner_list = []
    for i in range(N):
        inner_list.append([inner_pos[i][0], inner_pos[i][1], inner_angles[i]])
        
    return inner_list, [center[0], center[1]], side, outer_angle
# EVOLVE-BLOCK-END
