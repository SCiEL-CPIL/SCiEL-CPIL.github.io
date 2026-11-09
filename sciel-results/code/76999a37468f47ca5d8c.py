import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Outer hexagon normals (unit vectors perpendicular to sides)
    # Outer hexagon is fixed at angle 0, center (0,0)
    # Sides are at angles 0, 60, 120, 180, 240, 300
    # Normals point outward
    angles = np.deg2rad(np.arange(0, 360, 60))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Function to get vertices of a unit hexagon
    def get_vertices(cx, cy, theta_rad):
        # Vertices of a unit hexagon centered at origin, angle 0
        # Side length 1. Distance from center to vertex is 1.
        # Vertices at angles 0, 60, 120, 180, 240, 300
        v_angles = np.deg2rad(np.arange(0, 360, 60)) + theta_rad
        v_x = np.cos(v_angles)
        v_y = np.sin(v_angles)
        return np.column_stack((v_x + cx, v_y + cy))

    # Objective function: returns the minimum side length of outer hexagon
    def objective(params):
        # params: [x0, y0, theta0, x1, y1, theta1, ...]
        max_proj = 0.0
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            theta = params[3*i+2]
            
            verts = get_vertices(cx, cy, theta)
            # Project vertices onto normals
            # proj = verts @ normals.T -> shape (6, 6)
            projs = verts @ normals.T
            # The constraint is proj <= s * sqrt(3) / 2
            # So s >= max(proj) / (sqrt(3)/2)
            local_max = np.max(projs)
            if local_max > max_proj:
                max_proj = local_max
        
        return max_proj / (np.sqrt(3) / 2)

    # Initial guess
    # Create a flower pattern for 7, and place 4 more randomly or in a ring
    def initial_guess():
        params = np.zeros(3*N)
        # Center
        params[0:3] = [0.0, 0.0, 0.0]
        
        # 6 neighbors
        dist = np.sqrt(3)
        for k in range(6):
            ang = k * np.pi / 3
            idx = 3 * (k + 1)
            params[idx] = dist * np.cos(ang)
            params[idx+1] = dist * np.sin(ang)
            params[idx+2] = 0.0
            
        # 4 more: place them further out
        # Let's put them at distance 2*dist in directions 0, pi/2, pi, 3pi/2?
        # Or just randomize them a bit to allow optimizer to find a better spot
        for k in range(4):
            idx = 3 * (k + 7)
            ang = k * np.pi / 2
            r = 2.5 * dist
            params[idx] = r * np.cos(ang)
            params[idx+1] = r * np.sin(ang)
            params[idx+2] = 0.0
            
        return params

    # Run optimization
    best_result = None
    best_val = np.inf
    
    # Try a few different initializations to avoid local minima
    inits = [initial_guess()]
    
    # Add a random initialization
    rand_params = np.random.randn(3*N) * 5.0
    inits.append(rand_params)
    
    # Add a grid-like initialization
    grid_params = np.zeros(3*N)
    positions = []
    # Generate a hexagonal grid of points
    # Just use a square grid for simplicity in init
    for i in range(N):
        row = i // 4
        col = i % 4
        x = (col - 1.5) * 2.0
        y = (row - 1.0) * 1.8
        grid_params[3*i] = x
        grid_params[3*i+1] = y
        grid_params[3*i+2] = 0.0
    inits.append(grid_params)

    for x0 in inits:
        try:
            # Use Powell method, which is good for non-smooth problems
            result = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'disp': False})
            if result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract results
    params = best_result.x
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        theta_deg = np.rad2deg(params[3*i+2])
        inner.append([x, y, theta_deg])
    
    side_length = best_result.fun
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side_length, outer_angle
# EVOLVE-BLOCK-END
