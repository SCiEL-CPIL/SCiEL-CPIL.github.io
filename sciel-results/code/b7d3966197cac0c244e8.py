import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Directions for hexagon sides (normals)
    # u0: (1, 0)
    # u1: (0.5, sqrt(3)/2)
    # u2: (-0.5, sqrt(3)/2)
    u0 = np.array([1.0, 0.0])
    u1 = np.array([0.5, np.sqrt(3.0)/2.0])
    u2 = np.array([-0.5, np.sqrt(3.0)/2.0])
    us = np.array([u0, u1, u2])
    
    def calculate_penalty(W, centers):
        # centers: (N, 2) array
        penalty = 0.0
        
        # Containment penalty
        # max_k |dot(c_i, u_k)| <= W - 1
        # violation = max_k |dot(c_i, u_k)| - (W - 1)
        dots = np.abs(centers @ us.T) # (N, 3)
        max_dots = np.max(dots, axis=1) # (N,)
        violations_cont = np.maximum(0.0, max_dots - (W - 1.0))
        penalty += np.sum(violations_cont**2)
        
        # Separation penalty
        # For each pair (i, j), max_k |dot(c_i - c_j, u_k)| >= 2
        # violation = max(0, 2 - max_k |dot(d, u_k)|)
        # We can vectorize this partially
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dots_d = np.abs(d @ us.T)
                max_dot_d = np.max(dots_d)
                viol = max(0.0, 2.0 - max_dot_d)
                penalty += viol**2
                
        return penalty

    def objective(vars):
        W = vars[0]
        centers = vars[1:].reshape(N, 2)
        
        # Soft constraint on W to prevent it from going to 0
        # W must be at least 2 (for 1 hex)
        if W < 2.0:
            W_obj = 2.0 + (2.0 - W)**2
        else:
            W_obj = W
            
        pen = calculate_penalty(W, centers)
        
        return W_obj + 10.0 * pen

    def get_initial_guess(seed):
        np.random.seed(seed)
        # Hexagonal lattice initialization
        # Lattice vectors for unit hex packing (center distance 2)
        # a1 = (3, 0)
        # a2 = (1.5, 3*sqrt(3)/2)
        a1 = np.array([3.0, 0.0])
        a2 = np.array([1.5, 3.0*np.sqrt(3.0)/2.0])
        
        pts = []
        # Generate a grid of points
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i*a1 + j*a2
                pts.append(p)
        
        pts = np.array(pts)
        # Shuffle and take N
        idx = np.random.choice(len(pts), N, replace=False)
        centers = pts[idx]
        
        # Add some random noise
        centers += np.random.normal(0, 0.5, size=(N, 2))
        
        # Initial W estimate
        max_ext = np.max(np.abs(centers @ us.T))
        W_init = max_ext + 1.0 + 1.0
        
        return np.concatenate(([W_init], centers.flatten()))

    best_W = float('inf')
    best_centers = None
    
    # Multi-start optimization
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        x0 = get_initial_guess(k)
        
        try:
            res = minimize(objective, x0, method='COBYLA', 
                           options={'maxiter': 500, 'rhobeg': 1.0, 'catol': 1e-6})
            
            if res.success or res.fun < best_W:
                W_opt = res.x[0]
                c_opt = res.x[1:].reshape(N, 2)
                
                # Check validity
                # 1. Containment
                dots = np.abs(c_opt @ us.T)
                max_dots = np.max(dots, axis=1)
                if np.any(max_dots > W_opt - 1.0 + 1e-5):
                    continue
                    
                # 2. Separation
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = c_opt[i] - c_opt[j]
                        dots_d = np.abs(d @ us.T)
                        max_dot_d = np.max(dots_d)
                        if max_dot_d < 2.0 - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and W_opt < best_W:
                    best_W = W_opt
                    best_centers = c_opt
                    
        except Exception:
            continue

    if best_centers is None:
        # Fallback to a simple grid if optimization fails
        # This should not happen
        best_W = 10.0
        best_centers = np.zeros((N, 2))
        # Place them in a line
        for i in range(N):
            best_centers[i] = [3.0*i, 0.0]

    # Construct return values
    # inner: rows (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), 0.0])
    
    # Outer hexagon: center (0,0), side best_W, angle 0
    return inner, (0.0, 0.0), float(best_W), 0.0

# EVOLVE-BLOCK-END
