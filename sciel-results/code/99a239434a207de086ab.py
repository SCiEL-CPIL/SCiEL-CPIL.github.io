import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Helper to generate vertices of a unit hexagon at (x, y) with angle 0
    def get_hex_vertices(x, y):
        angles = np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([
            x + np.cos(angles),
            y + np.sin(angles)
        ])

    # Helper to calculate minimal outer hexagon side length for a set of points
    def calc_min_outer_L(points):
        # points: (M, 2) array
        # We optimize outer center (cx, cy) and angle theta
        # Objective: Max over all points of required L
        
        def objective_outer(params):
            cx, cy, theta = params
            # Vectors from center to points
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            r = np.sqrt(dx**2 + dy**2)
            phi = np.arctan2(dy, dx)
            
            # Hexagon normals are at theta + k * 60
            # For each point, find the closest normal
            # delta = (phi - theta) % (pi/3) - pi/6
            # We need to handle the modulo correctly for vectorization
            # phi is in [-pi, pi]
            # theta is in [0, 2pi]
            
            # Relative angle to the hexagon grid
            rel = (phi - theta) % (np.pi / 3)
            # Closest normal is at 0 or pi/3?
            # The distance to edge is A / cos(delta) where delta is angle from normal.
            # Normal is at 0. Edge is at 0.
            # If rel is close to 0, delta is small.
            # If rel is close to pi/3, delta is close to pi/3 (which is 30 deg).
            # So delta = min(rel, pi/3 - rel) ?
            # Actually, the function is symmetric around pi/6? No.
            # The normal is at 0. The vertex of the hexagon is at pi/6 (30 deg).
            # So the range of "closest normal" is [-pi/6, pi/6].
            # rel is in [0, pi/3).
            # If rel < pi/6, delta = rel.
            # If rel >= pi/6, delta = pi/3 - rel.
            
            delta = np.minimum(rel, np.pi / 3 - rel)
            
            # Required L for each point:
            # L * sqrt(3)/2 >= r * cos(delta)
            # L >= 2 * r * cos(delta) / sqrt(3)
            L_req = 2 * r * np.cos(delta) / np.sqrt(3)
            
            return np.max(L_req)

        # Initial guess for outer hexagon
        # Center of mass of points
        cx0, cy0 = np.mean(points, axis=0)
        theta0 = 0.0
        
        # Optimize outer hexagon
        res = minimize(objective_outer, [cx0, cy0, theta0], method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500})
        
        return res.fun

    # Initial positions: 7-cluster + 5 random
    # 7-cluster centers
    centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0.0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    
    # Add 5 more, placed roughly outside
    # (2*sqrt(3), 0), (2*sqrt(3)/2 + sqrt(3)/2, 1.5+1.5) ... just random near
    np.random.seed(42)
    for _ in range(5):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(2.5, 4.0)
        centers.append((dist * np.cos(angle), dist * np.sin(angle)))
    
    initial_pos = np.array(centers).flatten()
    
    # Objective for the main optimizer: rearrange inner hexagons
    def objective_inner(params):
        # params: 24 floats (12 x, 12 y)
        pos = params.reshape(12, 2)
        
        # Generate all vertices
        vertices = np.vstack([get_hex_vertices(x, y) for x, y in pos])
        
        # Calculate minimal outer L
        L = calc_min_outer_L(vertices)
        
        return L

    # Run optimization
    # Nelder-Mead is good for this
    res = minimize(objective_inner, initial_pos, method='Nelder-Mead', 
                   options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 2000})
    
    best_pos = res.x.reshape(12, 2)
    
    # Get the final optimal outer hexagon parameters
    vertices = np.vstack([get_hex_vertices(x, y) for x, y in best_pos])
    # Re-run the outer optimization to get the exact center and angle
    def objective_outer_full(params):
        cx, cy, theta = params
        dx = vertices[:, 0] - cx
        dy = vertices[:, 1] - cy
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        rel = (phi - theta) % (np.pi / 3)
        delta = np.minimum(rel, np.pi / 3 - rel)
        L_req = 2 * r * np.cos(delta) / np.sqrt(3)
        return np.max(L_req)

    cx0, cy0 = np.mean(vertices, axis=0)
    res_outer = minimize(objective_outer_full, [cx0, cy0, 0.0], method='Nelder-Mead', 
                         options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
    
    final_cx, final_cy, final_theta = res_outer.x
    final_L = res_outer.fun
    
    # Convert theta to degrees
    final_theta_deg = np.rad2deg(final_theta)
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    # Since we assumed angle 0 for inner hexagons in the optimization,
    # we return 0.0 for angle.
    inner = [[float(x), float(y), 0.0] for x, y in best_pos]
    
    return inner, (float(final_cx), float(final_cy)), float(final_L), float(final_theta_deg)

# EVOLVE-BLOCK-END
