import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Normal vectors for the outer hexagon (side S, rot 0)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Support function for a unit hexagon (flat top, rot 0)
    # Vertices at angles 0, 60, ...
    # h(phi) = max(v_k . n)
    # We need h(alpha_k - theta)
    # Precompute vertices of unit hexagon
    hex_verts = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)])
    
    def get_S_min(p, theta):
        # p: (2,) array
        # theta: scalar
        # Returns minimal S for this hexagon to fit
        # h_k = max (R(theta) * v_k) . normals[k]
        # Actually, h(alpha, theta) = max (v_k . R(-theta) . n_alpha)
        # Which is max ( (R(theta) v_k) . n_alpha )
        
        # Rotate vertices
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_verts = hex_verts @ R.T
        
        # Dot products with normals
        # rot_verts shape (6, 2), normals shape (6, 2)
        # We want max over vertices for each normal
        # dots[j, k] = rot_verts[j] . normals[k]
        dots = rot_verts @ normals.T  # (6, 6)
        h_vals = np.max(dots, axis=0)  # (6,)
        
        # Constraints: h_vals[k] + p . normals[k] <= S * sqrt(3)/2
        # S >= (h_vals[k] + p . normals[k]) * 2 / sqrt(3)
        p_dots = p @ normals.T  # (6,)
        S_req = (h_vals + p_dots) * 2.0 / np.sqrt(3.0)
        return np.max(S_req)
    
    def objective_and_grad(x):
        # x: [x1, y1, th1, ..., x12, y12, th12]
        # Fix center of mass to 0
        # We will handle this by projecting or just adding a penalty
        # Simpler: just add penalty for center of mass != 0
        
        if time.time() > deadline:
            return 1e10, None
            
        xs = x[0::3]
        ys = x[1::3]
        ths = x[2::3]
        
        # Center of mass penalty
        cm_x = np.mean(xs)
        cm_y = np.mean(ys)
        cm_pen = (cm_x**2 + cm_y**2) * 100.0
        
        # Calculate S_min for each
        S_mins = np.zeros(N)
        for i in range(N):
            S_mins[i] = get_S_min(np.array([xs[i], ys[i]]), ths[i])
            
        # Soft max for S_total
        # Use log-sum-exp for stability
        max_S = np.max(S_mins)
        soft_S = max_S + np.log(np.sum(np.exp(S_mins - max_S))) / N
        # Or just use max for simplicity if it converges, but soft is better for gradients
        # Let's use a smooth approximation
        tau = 0.1
        soft_S = -tau * np.log(np.mean(np.exp(-S_mins / tau)))
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # Min distance for disjoint unit hexagons is approx sqrt(3) if aligned
                # If we allow rotation, it can be less, but sqrt(3) is a safe bound
                # Using a slightly larger value ensures no overlap in initial phases
                min_dist = 1.732  # sqrt(3)
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2 * 10.0
                    
        total_cost = soft_S + overlap_pen + cm_pen
        return total_cost, None

    # Initial guess: 3x4 grid
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    init_x = []
    for r in range(3):
        for c in range(4):
            p = c * v1 + r * v2
            # Center the grid
            # We will let the optimizer center it, but start near 0
            init_x.append(p[0])
            init_x.append(p[1])
            init_x.append(0.0)
            
    init_x = np.array(init_x)
    
    # Optimize
    best_x = init_x
    best_cost = objective_and_grad(init_x)[0]
    
    # Use L-BFGS-B
    # We might need multiple restarts or just one good run
    try:
        res = minimize(objective_and_grad, init_x, method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
        if time.time() < deadline:
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
    except Exception as e:
        pass

    # Extract best result
    xs = best_x[0::3]
    ys = best_x[1::3]
    ths = best_x[2::3]
    
    # Recalculate final S
    S_final = 0.0
    for i in range(N):
        S_final = max(S_final, get_S_min(np.array([xs[i], ys[i]]), ths[i]))
        
    inner = [[xs[i], ys[i], np.rad2deg(ths[i])] for i in range(N)]
    return inner, (0.0, 0.0), S_final, 0.0

