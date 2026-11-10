import numpy as np
import scipy.optimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590  # 10s margin

    # 1. Initial Guess
    # 7-hex cluster + 5 corners
    sqrt3 = np.sqrt(3)
    
    # Center
    centers = np.array([[0.0, 0.0]])
    
    # Ring 1 (6 hexes)
    # e1 = (sqrt3, 0), e2 = (sqrt3/2, 1.5), e3 = (-sqrt3/2, 1.5)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3/2, 1.5])
    e3 = e1 - e2 # (-sqrt3/2, 1.5)
    
    ring1 = [e1, e2, e3, -e1, -e2, -e3]
    
    # Ring 2 Corners (6 positions)
    # V1 = e1+e2, V2 = e2+e3, V3 = e3-e1, V4 = -e1-e2, V5 = -e2-e3, V6 = -e3+e1
    corners = [
        e1 + e2,
        e2 + e3,
        e3 - e1,
        -e1 - e2,
        -e2 - e3,
        -e3 + e1
    ]
    
    # Select 5 corners
    selected_corners = corners[:5]
    
    # Combine
    all_centers = np.vstack([centers, np.array(ring1), np.array(selected_corners)])
    
    # Flatten for optimizer
    x0 = all_centers.flatten()
    
    # Outer angle initial guess
    theta0 = 0.0
    x0 = np.append(x0, theta0)
    
    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        centers = params[:24].reshape(12, 2)
        theta = params[24]
        
        # Calculate centroid
        centroid = np.mean(centers, axis=0)
        shifted_centers = centers - centroid
        
        # Outer hexagon normals
        # Normals at angles theta, theta+60, ...
        angles = theta + np.deg2rad(60 * np.arange(6))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_proj = 0.0
        
        # For each inner hexagon
        for i in range(12):
            cx, cy = shifted_centers[i]
            # Inner hexagon vertices (side 1, angle 0)
            # Vertices of unit hexagon centered at (0,0) with angle 0:
            # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
            # We can pre-compute these
            v_angles = np.deg2rad(60 * np.arange(6))
            verts = np.column_stack([np.cos(v_angles), np.sin(v_angles)])
            
            # Translate to (cx, cy)
            verts = verts + np.array([cx, cy])
            
            # Project onto normals
            # proj = verts @ normals.T  -> shape (6, 6)
            proj = verts @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # s = (2/sqrt3) * max_proj
        s = (2.0 / sqrt3) * max_proj
        return s

    # Optimize
    best_x = x0
    best_f = objective(x0)
    
    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except Exception as e:
        pass
        
    # Extract best solution
    centers = best_x[:24].reshape(12, 2)
    theta_deg = np.rad2deg(best_x[24])
    
    # Recalculate centroid for final output
    centroid = np.mean(centers, axis=0)
    
    # Inner hexagon data: (x, y, angle_deg)
    # Angle is 0 for all
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = centers[:, 0]
    inner_data[:, 1] = centers[:, 1]
    inner_data[:, 2] = 0.0
    
    # Outer hexagon data: (x, y, angle_deg)
    outer_data = np.array([centroid[0], centroid[1], theta_deg])
    
    # Calculate s
    s = best_f
    
    # Safety check: ensure s is finite and valid
    if not np.isfinite(s) or s < np.sqrt(12):
        # Fallback to a very large hexagon if optimization failed
        s = 10.0
        outer_data = np.array([0.0, 0.0, 0.0])
        # Reset centers to initial guess
        centers = x0[:24].reshape(12, 2)
        inner_data[:, 0] = centers[:, 0]
        inner_data[:, 1] = centers[:, 1]
        
    return inner_data, outer_data, s

if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"s: {s}")
