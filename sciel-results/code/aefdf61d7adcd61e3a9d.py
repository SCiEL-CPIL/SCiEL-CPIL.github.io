import numpy as np
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    def compute_distances(points):
        # points is (16, 2)
        # Return pairwise distances (16, 16) or (120,)
        # Using pdist is efficient
        from scipy.spatial.distance import pdist
        return pdist(points)

    def objective(params):
        # params is flat array of 28 variables (14 points x 2 coords)
        # p0 = (0,0), p1 = (1,0) are fixed
        p_rest = params.reshape(14, 2)
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        points = np.vstack([p0, p1, p_rest])
        
        dists = compute_distances(points)
        
        # Soft min and max
        # beta controls sharpness
        beta = 20.0
        
        # Soft max: (1/beta) * log(sum(exp(beta * d)))
        # Soft min: -(1/beta) * log(sum(exp(-beta * d)))
        
        # Numerical stability: subtract max for exp
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # log(sum(exp(beta * d))) = beta * max_d + log(sum(exp(beta * (d - max_d))))
        log_sum_exp_max = beta * max_d + np.log(np.sum(np.exp(beta * (dists - max_d))))
        soft_max = log_sum_exp_max / beta
        
        # log(sum(exp(-beta * d))) = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
        # Note: exp(-beta * (d - min_d)) = exp(-beta * d + beta * min_d)
        # Sum = exp(beta * min_d) * sum(exp(-beta * d))
        # So log(sum(exp(-beta * d))) = beta * min_d + log(sum(exp(-beta * (d - min_d))))
        log_sum_exp_min = -beta * min_d + np.log(np.sum(np.exp(-beta * (dists - min_d))))
        # Wait, soft_min = - (1/beta) * log(sum(exp(-beta * d)))
        # log(sum(exp(-beta * d))) = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
        # So soft_min = - (1/beta) * (-beta * min_d + log_sum_exp_min_part)
        # soft_min = min_d - (1/beta) * log(sum(exp(-beta * (d - min_d))))
        
        # Let's re-calculate carefully
        # S_max = sum(exp(beta * d))
        # log(S_max) = beta * max_d + log(sum(exp(beta * (d - max_d))))
        # soft_max = log(S_max) / beta
        
        # S_min = sum(exp(-beta * d))
        # log(S_min) = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
        # soft_min = - log(S_min) / beta
        
        log_s_max = beta * max_d + np.log(np.sum(np.exp(beta * (dists - max_d))))
        soft_max = log_s_max / beta
        
        log_s_min = -beta * min_d + np.log(np.sum(np.exp(-beta * (dists - min_d))))
        soft_min = -log_s_min / beta
        
        if soft_min <= 0:
            return 1e10
            
        return np.log(soft_max) - np.log(soft_min)

    def run_optimization(init_points):
        # init_points is (16, 2)
        # Fix p0, p1
        # Check if init points are close to (0,0) and (1,0)
        # We will just use the last 14 points as initial guess for params
        # But we should ensure the fixed points are consistent with the init
        # For simplicity, we will just take the last 14 points from a valid init
        # where p0=(0,0) and p1=(1,0) are already set.
        
        p0 = init_points[0]
        p1 = init_points[1]
        # We assume the init points are already translated/rotated such that p0=(0,0) and p1=(1,0)
        # If not, we should transform them.
        
        # Transform init points to fixed frame
        # Translate p0 to origin
        pts = init_points - p0
        # Rotate so p1 is on x-axis
        p1_trans = pts[1]
        norm = np.linalg.norm(p1_trans)
        if norm < 1e-9:
            return None
        # Rotation matrix
        c = p1_trans[0] / norm
        s = p1_trans[1] / norm
        R = np.array([[c, -s], [s, c]])
        # Rotate all points
        pts_rot = pts @ R.T
        # Scale so p1 is at (1,0)
        # p1_rot should be (norm, 0)
        pts_scaled = pts_rot / norm
        
        # Now pts_scaled[0] is (0,0) and pts_scaled[1] is (1,0)
        # Extract params
        params0 = pts_scaled[2:].flatten()
        
        res = minimize(objective, params0, method='Powell', options={'maxiter': 1000, 'disp': False})
        
        if res.success or res.fun < 1e9:
            best_params = res.x
            p_rest = best_params.reshape(14, 2)
            p0_final = np.array([0.0, 0.0])
            p1_final = np.array([1.0, 0.0])
            final_points = np.vstack([p0_final, p1_final, p_rest])
            return final_points
        return None

    best_points = None
    best_score = -1e10

    # Generate initializations
    # 1. Hexagonal
    def hex_init():
        pts = []
        pts.append([0.0, 0.0])
        # Ring 1: 6 points
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2: Need 9 more points.
        # Standard hex ring 2 has 12 points. We pick 9.
        # To maintain symmetry, maybe pick every 4th? Or just 9 consecutive?
        # Let's try 9 consecutive in ring 2 (distance 2)
        for i in range(9):
            angle = (i + 0.5) * np.pi / 6 # Offset?
            # Actually, hex lattice ring 2 points are at angles 0, 30, 60... 
            # Distance 2 points:
            # (2,0), (1, sqrt(3)), etc.
            # Let's just use a grid and pick 16 closest to origin?
            pass
        
        # Simpler: use a grid
        return None

    # Let's use a grid init
    def grid_init():
        pts = []
        # 4x4 grid
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts)
        return pts

    def random_init():
        np.random.seed(42)
        return np.random.randn(16, 2)

    inits = []
    inits.append(grid_init())
    inits.append(random_init())
    
    # Hex-like init
    hex_pts = []
    hex_pts.append([0,0])
    for i in range(6):
        hex_pts.append([np.cos(i*np.pi/3), np.sin(i*np.pi/3)])
    # Add 9 points in a ring of radius 2
    for i in range(9):
        angle = i * 2 * np.pi / 9
        hex_pts.append([2*np.cos(angle), 2*np.sin(angle)])
    inits.append(np.array(hex_pts))

    for init in inits:
        if time.time() > deadline:
            break
        result = run_optimization(init)
        if result is not None:
            # Calculate score
            dists = compute_distances(result)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max > 0:
                score = (d_min / d_max) ** 2
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_points = result

    if best_points is None:
        # Fallback
        return np.random.randn(16, 2)

    return best_points
