import numpy as np
from scipy.optimize import minimize, brentq
from scipy.spatial.distance import pdist

def construct():
    # We optimize 7 parameters u_1 ... u_7
    # w_i = exp(u_i) for i=1..7, w_8 = 1
    # s_i = w_i / sum(w)
    
    def get_side_lengths(u):
        # u is array of 7 floats
        w = np.exp(u)
        w = np.append(w, 1.0)
        s = w / np.sum(w)
        return s

    def find_radius(s):
        # Find R such that sum(2 * arcsin(s_i / (2R))) = 2 * pi
        # R must be > max(s)/2
        R_min = np.max(s) / 2.0
        
        def f(R):
            # Avoid division by zero or invalid arcsin
            # R > R_min ensures s_i / (2R) < 1
            args = s / (2.0 * R)
            # Clip to [0, 1] just in case of numerical errors, though R > R_min should prevent >1
            args = np.clip(args, 0.0, 1.0)
            angles = 2.0 * np.arcsin(args)
            return np.sum(angles) - 2.0 * np.pi

        # Check if solution exists
        # f(R_min) should be > 0 for valid polygon
        # If f(R_min) <= 0, it means the polygon is degenerate or invalid
        f_min = f(R_min + 1e-12)
        if f_min < 0:
            return None
            
        # Find upper bound
        R_high = R_min * 10
        while f(R_high) > 0:
            R_high *= 2
            if R_high > 1e6:
                return None
        
        try:
            R = brentq(f, R_min + 1e-12, R_high, xtol=1e-12)
            return R
        except:
            return None

    def objective(u):
        s = get_side_lengths(u)
        
        # Check polygon inequality
        max_s = np.max(s)
        sum_others = np.sum(s) - max_s
        if max_s >= sum_others:
            return 1e10
            
        R = find_radius(s)
        if R is None:
            return 1e10
            
        # Compute angles
        args = s / (2.0 * R)
        args = np.clip(args, 0.0, 1.0)
        angles = 2.0 * np.arcsin(args)
        
        # Compute vertices
        # Start at angle 0
        cum_angles = np.cumsum(angles)
        # The last angle should bring us back to 2*pi (mod 2*pi)
        # vertices are at cum_angles[0:-1] and 0? 
        # Vertex i is at angle sum_{j=1}^i angles[j]
        # V_0 at 0
        # V_1 at angles[0]
        # ...
        # V_7 at sum(angles[0:7])
        # V_8 = V_0
        
        theta = np.concatenate(([0.0], cum_angles[:-1]))
        # Wait, cum_angles has 8 elements.
        # theta[0] = 0
        # theta[1] = angles[0]
        # ...
        # theta[7] = sum(angles[0:7])
        # The distance from V_7 to V_0 corresponds to angle 2*pi - theta[7] = angles[7]
        # This is consistent.
        
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        points = np.column_stack((x, y))
        
        # Scale to diameter 2
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e10
        points = points * (2.0 / d_max)
        dists_scaled = dists * (2.0 / d_max)
        
        # Log product
        log_prod = np.sum(np.log(dists_scaled + 1e-12))
        
        return -log_prod

    # Initial guess: Regular Octagon
    # w_i all equal => u_i all equal
    u0 = np.zeros(7)
    
    # Optimize
    result = minimize(objective, u0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    u_opt = result.x
    s = get_side_lengths(u_opt)
    R = find_radius(s)
    
    if R is None:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.column_stack((x, y)).tolist()
        
    args = s / (2.0 * R)
    args = np.clip(args, 0.0, 1.0)
    angles = 2.0 * np.arcsin(args)
    cum_angles = np.cumsum(angles)
    theta = np.concatenate(([0.0], cum_angles[:-1]))
    
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    points = np.column_stack((x, y))
    
    # Final scaling
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points * (2.0 / d_max)
        
    return points.tolist()
