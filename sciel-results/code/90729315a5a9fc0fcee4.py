import numpy as np
import time

N = 11

def h_unit(alpha):
    # Support function of unit hexagon (side 1, vertex at distance 1)
    # alpha is angle
    # h(alpha) = max(cos(alpha - k*pi/3)) for k=0..5
    # This assumes vertices at 0, 60, ...
    # Side length 1 means distance from center to vertex is 1.
    # Distance from center to side is sqrt(3)/2.
    # This matches the task "unit regular hexagons of side length 1".
    k = np.arange(6) * np.pi / 3
    return np.max(np.cos(alpha - k))

def calculate_side_length(centers, angles):
    # centers: (N, 2)
    # angles: (N,)
    # Returns minimum S
    alphas = np.linspace(0, 2*np.pi, 720, endpoint=False)
    h_cont_vals = h_unit(alphas) # h_unit is for S=1
    # h_cont(alpha) = S * h_unit(alpha)
    # Constraint: c_i . u(alpha) + h_unit(alpha - phi_i) <= S * h_unit(alpha)
    # S >= (c_i . u(alpha) + h_unit(alpha - phi_i)) / h_unit(alpha)
    
    max_s = 0.0
    for i in range(N):
        cx, cy = centers[i]
        phi = angles[i]
        u = np.array([np.cos(alphas), np.sin(alphas)])
        proj = cx * np.cos(alphas) + cy * np.sin(alphas)
        h_in = h_unit(alphas - phi)
        s_req = (proj + h_in) / h_cont_vals
        # Handle division by zero if h_cont is 0 (should not happen for hexagon)
        s_req = np.where(h_cont_vals > 1e-9, s_req, 0)
        max_s = max(max_s, np.max(s_req))
    return max_s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial lattice packing
    # 1-2-3-2-1 = 9 points. Add 2 more.
    # Lattice spacing sqrt(3)
    sp = np.sqrt(3)
    centers = []
    angles = np.zeros(N)
    
    # Generate lattice points
    # Use axial coordinates
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Convert axial to cartesian
            x = sp * (q + r/2.0)
            y = sp * (r * np.sqrt(3)/2.0)
            centers.append([x, y])
            
    # We have 25 points. We need 11.
    # Select the 11 closest to origin
    centers = np.array(centers)
    dists = np.linalg.norm(centers, axis=1)
    idx = np.argsort(dists)
    centers = centers[idx[:N]]
    
    # Optimize
    best_s = calculate_side_length(centers, angles)
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    # Simple local search / Nelder-Mead
    # Flatten parameters
    params = np.concatenate([centers.ravel(), angles])
    
    def objective(p):
        c = p[:2*N].reshape(N, 2)
        a = p[2*N:]
        return calculate_side_length(c, a)
        
    # Run a few random restarts with local optimization
    from scipy.optimize import minimize
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            p0 = params.copy()
        else:
            # Perturb initial lattice
            p0 = params + np.random.normal(0, 0.5, size=params.shape)
            
        try:
            res = minimize(objective, p0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_s:
                best_s = res.fun
                best_centers = res.x[:2*N].reshape(N, 2)
                best_angles = res.x[2*N:]
        except:
            pass
            
    # Return best result
    # The function returns (inner, center, side, angle)
    # inner: list of (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], np.degrees(best_angles[i])])
        
    return inner, (0.0, 0.0), best_s, 0.0

# EVOLVE-BLOCK-START
# (The above code is the implementation)
# EVOLVE-BLOCK-END
