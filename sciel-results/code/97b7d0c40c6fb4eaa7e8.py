import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 550  # Leave 50s margin
    
    # Initial configurations
    inits = []
    
    # 1. Baseline Grid
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers0 = np.stack([cx, cy], axis=1)
    sides0 = np.full(n, side)
    inits.append((centers0, sides0))
    
    # 2. "3.2 Heuristic" (2x2 block of 0.3, 4x0.4 strips, 2x0.2 corners)
    # This has overlaps, but is a good starting point for optimizer
    c1 = np.array([
        [0.15, 0.15], [0.45, 0.15], [0.15, 0.45], [0.45, 0.45], # 2x2 of 0.3
        [0.8, 0.2], [0.8, 0.6], # 2x 0.4 in right strip (x=0.6-1.0)
        [0.2, 0.8], [0.6, 0.8], # 2x 0.4 in top strip (y=0.6-1.0)
        [0.9, 0.9], # 0.2 in corner
        [0.7, 0.9]  # 0.2 next to corner
    ])
    s1 = np.array([
        0.3, 0.3, 0.3, 0.3,
        0.4, 0.4,
        0.4, 0.4,
        0.2,
        0.2
    ])
    inits.append((c1, s1))
    
    # 3. 3x3 Grid + 1 small
    s3 = 1.0/3.0
    c3 = np.array([
        [0.5*s3, 0.5*s3], [1.5*s3, 0.5*s3], [2.5*s3, 0.5*s3],
        [0.5*s3, 1.5*s3], [1.5*s3, 1.5*s3], [2.5*s3, 1.5*s3],
        [0.5*s3, 2.5*s3], [1.5*s3, 2.5*s3], [2.5*s3, 2.5*s3],
        [0.99, 0.99]
    ])
    s3_vec = np.array([s3]*9 + [0.01])
    inits.append((c3, s3_vec))

    best_centers = inits[0][0]
    best_sides = inits[0][1]
    best_score = np.sum(inits[0][1])
    
    def compute_penalty(centers, sides):
        # centers: (10, 2), sides: (10,)
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Bounds penalty
        # Square i is inside [0,1] if cx - s/2 >= 0 and cx + s/2 <= 1
        # Penalty: max(0, 0 - (cx - s/2)) + max(0, (cx + s/2) - 1) + ...
        left_viol = np.maximum(0, 0.0 - (cx - s/2.0))
        right_viol = np.maximum(0, (cx + s/2.0) - 1.0)
        bottom_viol = np.maximum(0, 0.0 - (cy - s/2.0))
        top_viol = np.maximum(0, (cy + s/2.0) - 1.0)
        
        bounds_pen = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                     np.sum(bottom_viol**2) + np.sum(top_viol**2)
        
        # Overlap penalty
        # For each pair i, j
        # Overlap if |cx_i - cx_j| < (s_i+s_j)/2 AND |cy_i - cy_j| < (s_i+s_j)/2
        # Distance metric:
        # dx = |cx_i - cx_j| - (s_i+s_j)/2
        # dy = |cy_i - cy_j| - (s_i+s_j)/2
        # If dx < 0 and dy < 0, they overlap.
        # Penalty: max(0, -dx)^2 + max(0, -dy)^2 ? 
        # Actually, if they overlap, we want to push them apart.
        # A common smooth penalty is:
        # If dx < 0 and dy < 0:
        #   pen = (max(0, -dx)**2 + max(0, -dy)**2)
        # This drives dx and dy to be >= 0.
        
        pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = np.abs(cx[i] - cx[j]) - 0.5 * (s[i] + s[j])
                dy = np.abs(cy[i] - cy[j]) - 0.5 * (s[i] + s[j])
                if dx < 0 and dy < 0:
                    pen += (-dx)**2 + (-dy)**2
        return bounds_pen + 10.0 * pen # Weight overlap more heavily
    
    def objective(params):
        # params: 20 values (10 cx, 10 cy, 10 s)
        cx = params[0:10]
        cy = params[10:20]
        s = params[20:30]
        
        # Ensure sides are positive
        s = np.maximum(s, 1e-6)
        
        centers = np.stack([cx, cy], axis=1)
        
        pen = compute_penalty(centers, s)
        sum_s = np.sum(s)
        
        # We want to maximize sum_s, so minimize -sum_s + pen
        # Use a large weight for penalty to ensure validity first
        return pen - 100.0 * sum_s

    best_valid_centers = None
    best_valid_sides = None
    best_valid_score = -np.inf

    for init_c, init_s in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Flatten
        x0 = np.concatenate([init_c[:, 0], init_c[:, 1], init_s])
        
        # Bounds for optimizer
        # cx, cy in [0, 1] (centers can go slightly outside if sides are small, but let's keep centers in [0,1])
        # Actually, if s=1, cx must be 0.5.
        # Let's allow cx in [0, 1] and s in [0, 1].
        bounds = [(0.0, 1.0)] * 10 + [(0.0, 1.0)] * 10 + [(1e-6, 1.0)] * 10
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Extract result
            cx = res.x[0:10]
            cy = res.x[10:20]
            s = res.x[20:30]
            
            centers = np.stack([cx, cy], axis=1)
            
            # Check validity with strict tolerance
            tol = 1e-7
            valid = True
            
            # Bounds
            if np.any(centers[:, 0] - s/2.0 < -tol) or np.any(centers[:, 0] + s/2.0 > 1.0 + tol):
                valid = False
            if np.any(centers[:, 1] - s/2.0 < -tol) or np.any(centers[:, 1] + s/2.0 > 1.0 + tol):
                valid = False
            
            # Overlaps
            for i in range(n):
                for j in range(i+1, n):
                    t = 0.5 * (s[i] + s[j]) - tol
                    if np.abs(centers[i, 0] - centers[j, 0]) < t and np.abs(centers[i, 1] - centers[j, 1]) < t:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                score = np.sum(s)
                if score > best_valid_score:
                    best_valid_score = score
                    best_valid_centers = centers.copy()
                    best_valid_sides = s.copy()
                    
        except Exception as e:
            pass

    # If no valid solution found via optimization, return the best valid initial (which is the baseline)
    if best_valid_centers is None:
        return inits[0]
    else:
        return best_valid_centers, best_valid_sides
