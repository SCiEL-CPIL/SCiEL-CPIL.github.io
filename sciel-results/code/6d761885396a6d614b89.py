import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    # Fibonacci sphere initialization for angles
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles_init = np.zeros((N, 2))
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        theta = np.arccos(cos_theta)
        phi = i * golden_angle
        angles_init[i, 0] = theta
        angles_init[i, 1] = phi
        
    def get_positions(angles, R):
        theta = angles[:, 0]
        phi = angles[:, 1]
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        return np.column_stack((x, y, z))
        
    def objective(params):
        angles_flat = params[:2*N]
        R_val = params[-1]
        
        # Bounds for R
        if R_val < 0.1 or R_val > 0.49:
            return 1e6
            
        angles = angles_flat.reshape(N, 2)
        centers = get_positions(angles, R_val)
        
        # Compute min pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        r_pair = min_pair_dist / 2.0
        
        r_wall = 0.5 - R_val
        score = min(r_pair, r_wall)
        
        return -score
        
    # Estimate initial R
    sqrt_pi_N = np.sqrt(np.pi / N)
    R_init = 0.5 / (1.0 + sqrt_pi_N)
    
    init_params = np.concatenate([angles_init.flatten(), [R_init]])
    
    # Optimize using Powell's method
    res = minimize(objective, init_params, method='Powell', options={'maxiter': 2000, 'disp': False, 'maxfev': 30000})
    
    best_angles = res.x[:2*N].reshape(N, 2)
    best_R = res.x[-1]
    
    # Clamp R
    best_R = np.clip(best_R, 0.1, 0.49)
    
    final_centers = get_positions(best_angles, best_R)
    
    return final_centers
