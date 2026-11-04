import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Initial guess: 11 hexagons in a 3x3 grid (minus 2 corners) or similar
    # Let's use a cluster of 7 and 4 on the sides
    # Center
    pos = np.array([0.0, 0.0])
    angles = np.array([0.0])
    
    # Ring 1: 6 neighbors
    for i in range(6):
        theta = i * np.pi / 3
        pos = np.vstack([pos, [np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta)]])
        angles = np.append(angles, 0.0)
        
    # Ring 2: 4 more
    # Place them at 30, 150, 210, 330 degrees, distance 3.0
    # Rotate them by 30 degrees to fit
    for i in range(4):
        theta = (2*i + 1) * np.pi / 6  # 30, 150, 210, 330
        pos = np.vstack([pos, [3.0 * np.cos(theta), 3.0 * np.sin(theta)]])
        angles = np.append(angles, np.pi / 6) # 30 degrees

    # Optimize
    # We want to minimize the side length of the bounding hexagon.
    # Bounding hexagon side S is the max over all vertices of the inner hexagons
    # of the support function in the 6 directions.
    
    # Define the 6 directions for the outer hexagon (flat-top aligned with inner?)
    # Let's assume outer hexagon is also aligned (rotation 0).
    # Directions: 0, 60, 120, 180, 240, 300 degrees.
    # Support function h(theta) = max_i (p_i . v_theta + r_i(theta))
    # where r_i(theta) is the support function of the unit hexagon rotated by angles[i].
    
    # Support function of a unit hexagon with rotation phi:
    # h_hex(theta, phi) = max_{k=0..5} (v_theta . (R_phi v_k))
    # v_k are the vertices of the unit hexagon.
    
    # Vertices of unit hexagon (flat-top, center 0,0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Wait, standard hexagon with side 1 has circumradius 1.
    # Vertices at angles 0, 60, ...
    v_hex = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])
    
    # Precompute support function for each inner hexagon
    # h_i(theta) = max_k (v_theta . (R_{angles[i]} v_hex[k]))
    # R_phi v = [cos(phi) v_x - sin(phi) v_y, sin(phi) v_x + cos(phi) v_y]
    
    # We can simplify: h_i(theta) = h_unit(theta - angles[i])
    # h_unit(alpha) = max_k (cos(alpha - k*pi/3) * 1) ... no
    # h_unit(alpha) = max_k (cos(alpha) cos(k*pi/3) - sin(alpha) sin(k*pi/3))
    # This is the support function of the unit hexagon.
    
    # Let's use a simple gradient-free optimization (Nelder-Mead) on the positions and angles.
    # Variables: x_i, y_i, angle_i for i=0..10. Total 33 variables.
    # Objective: S = max_{j=0..5} (max_i (x_i cos(theta_j) + y_i sin(theta_j) + h_unit(theta_j - angle_i)))
    # where theta_j = j * pi / 3.
    
    # To make it differentiable, we can use a soft-max or just evaluate the max.
    # Nelder-Mead works with non-differentiable functions.
    
    from scipy.optimize import minimize
    
    def cost(params):
        # params: [x0, y0, a0, x1, y1, a1, ...]
        x = params[0::3]
        y = params[1::3]
        a = params[2::3]
        
        # Calculate S
        S = 0.0
        for j in range(6):
            theta_j = j * np.pi / 3
            cos_t = np.cos(theta_j)
            sin_t = np.sin(theta_j)
            
            # Projection of centers
            proj_centers = x * cos_t + y * sin_t
            
            # Support function of each hexagon in direction theta_j
            # h_unit(theta_j - a)
            # h_unit(alpha) = max_k cos(alpha - k*pi/3)
            # We can compute this efficiently.
            # The max of cos(alpha - k*pi/3) is 1 if alpha is close to k*pi/3.
            # Actually, h_unit(alpha) = 1 / cos(pi/6) * cos(alpha - round(alpha/(pi/3))*pi/3) ?
            # No, the support function of a regular polygon is:
            # h(alpha) = r * cos(alpha - k*pi/n) where k is such that |alpha - k*pi/n| <= pi/n.
            # r=1, n=6.
            # So h(alpha) = cos(alpha - k*pi/6 * 2) ? No, vertices at k*pi/3.
            # The "flat" sides are at pi/6 + k*pi/3.
            # The support function is 1 / cos(pi/6) * cos(alpha - (pi/6 + k*pi/3)) for the flat side?
            # No, the vertices are the points.
            # h(alpha) = max_k (v_k . u_alpha)
            # v_k = (cos(k*pi/3), sin(k*pi/3))
            # u_alpha = (cos(alpha), sin(alpha))
            # v_k . u_alpha = cos(alpha - k*pi/3)
            # So h_unit(alpha) = max_k cos(alpha - k*pi/3)
            
            # We can compute this by finding the closest multiple of pi/3.
            # Let delta = alpha % (pi/3)
            # If delta < pi/6, h = cos(delta)
            # If delta > pi/6, h = cos(pi/3 - delta)
            
            alphas = theta_j - a
            # Normalize alphas to [0, 2*pi)
            alphas = alphas % (2 * np.pi)
            # Map to [0, pi/3)
            deltas = alphas % (np.pi / 3)
            # h = max(cos(deltas), cos(pi/3 - deltas))
            # Since cos is decreasing on [0, pi], and symmetric:
            # h = cos(min(deltas, pi/3 - deltas))
            
            h_vals = np.cos(np.minimum(deltas, np.pi / 3 - deltas))
            
            # Max projection
            max_proj = np.max(proj_centers + h_vals)
            S = max(S, max_proj)
            
        return S

    # Initial parameters
    init_params = []
    for i in range(N):
        init_params.extend([pos[i][0], pos[i][1], angles[i]])
        
    # Run optimization
    best_result = None
    best_cost = cost(init_params)
    
    # Use Nelder-Mead
    # Limit iterations to stay within time
    max_iter = 1000
    
    try:
        res = minimize(cost, init_params, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res.x
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline:
        # Return best found so far
        pass
        
    # Use best_result or init_params
    final_params = best_result if best_result is not None else np.array(init_params)
    
    # Extract final positions and angles
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i + 1]
        a_deg = np.degrees(final_params[3*i + 2])
        inner.append([x, y, a_deg])
        
    # Outer hexagon: center (0,0), side S, rotation 0
    # S is best_cost
    return inner, (0.0, 0.0), best_cost, 0.0

# EVOLVE-BLOCK-END
