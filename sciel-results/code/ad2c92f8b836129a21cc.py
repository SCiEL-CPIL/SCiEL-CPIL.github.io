import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12
# Time limit management
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin for 150s limit

def get_hex_vertices(angle_deg):
    """
    Returns the 6 vertices of a unit regular hexagon centered at origin,
    rotated by angle_deg.
    Vertices are at radius 1.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Vertices at 0, 60, ..., 300 degrees
    base_angles = np.arange(6) * np.pi / 3.0
    angles = base_angles + angle_rad
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def point_in_hexagon_dist(point, S, alpha_deg):
    """
    Returns the maximum distance by which 'point' violates the boundary of
    a regular hexagon of side S, centered at origin, with angle alpha_deg.
    Returns 0 if inside.
    """
    alpha_rad = np.deg2rad(alpha_deg)
    # Normals to the sides
    # Side midpoints are at angles alpha, alpha+60, ...
    # Normals are perpendicular to sides? No, normals point from center to side midpoint.
    # For a regular hexagon with vertex at angle 0, the side opposite is at 180.
    # The normal to the side between vertices at -30 and 30 is at 0.
    # Wait, standard orientation:
    # If vertices are at 0, 60, ..., then sides are perpendicular to 30, 90, ...?
    # Let's stick to the support function definition.
    # Distance from center to side is D = S * sqrt(3) / 2.
    # The normals to the sides are at angles: alpha, alpha + 60, ... 
    # This assumes 'alpha' is the angle of the normal to the first side.
    
    normals = np.column_stack((np.cos(alpha_rad + np.arange(6)*np.pi/3), 
                               np.sin(alpha_rad + np.arange(6)*np.pi/3)))
    
    D = S * np.sqrt(3) / 2.0
    
    # Project point onto normals
    projections = point @ normals.T
    violations = projections - D
    return np.max(violations)

def cost_function(params, inner_verts):
    """
    params: [x1, y1, ..., x12, y12, S, alpha, theta]
    inner_verts: Pre-computed vertices for unit hex at angle 0? 
    No, theta is in params, so we need to rotate.
    Actually, to speed up, we can pre-compute rotation matrices or just use cos/sin.
    """
    # Time check
    if time.time() - START_TIME > TIME_LIMIT:
        return float('inf')

    # Unpack
    positions = params[:N*2].reshape(N, 2)
    S = params[N*2]
    alpha = params[N*2+1]
    theta = params[N*2+2]
    
    # Ensure S is positive
    if S < 0:
        S = 0.1

    cost = S
    
    # 1. Containment Penalty
    # Get vertices for current theta
    # We can compute vertices once per theta? No, theta is a variable.
    # But we can compute the 6 unit vertices once, and rotate them by theta.
    # Unit vertices at angle 0:
    # v_k = (cos(k*pi/3), sin(k*pi/3))
    base_verts = np.column_stack((np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)))
    
    # Rotate by theta
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    R = np.array([[c_theta, -s_theta], [s_theta, c_theta]])
    verts = base_verts @ R.T  # Shape (6, 2)
    
    for i in range(N):
        c_i = positions[i]
        # Vertices of inner hex i
        v_i = c_i + verts
        
        # Check each vertex against outer hex
        # point_in_hexagon_dist expects a single point? 
        # Let's vectorize:
        # Projections of all vertices onto normals
        # normals shape (6, 2), v_i shape (6, 2)
        # We need v_i @ normals.T -> (6, 6)
        # Max over columns (normals) for each row (vertex)?
        # No, for each vertex, check max violation over all 6 sides.
        # proj = v_i @ normals.T (6x6 matrix, proj[v][n])
        # violation[v][n] = proj[v][n] - D
        # max_violation[v] = max_n(violation[v][n])
        # total_violation = sum(max(0, max_violation[v]))
        
        proj = v_i @ normals.T
        D = S * np.sqrt(3) / 2.0
        violations = proj - D
        max_violations = np.max(violations, axis=1)
        # Sum of positive violations
        cost += np.sum(np.maximum(0, max_violations)) ** 2

    # 2. Overlap Penalty
    # Min distance sqrt(3)
    min_dist = np.sqrt(3)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                cost += (min_dist - d) ** 2

    return cost

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess: Cluster of 12
    # 1 center, 6 neighbors, 5 more
    # Neighbors at distance sqrt(3)
    d = np.sqrt(3)
    init_pos = []
    init_pos.append([0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3.0
        init_pos.append([d * np.cos(angle), d * np.sin(angle)])
    # Add 5 more in the next ring
    # Next ring distance is 2*d? Or d * sqrt(3)?
    # In hex lattice, next ring is at distance 2*sqrt(3) for axial, or sqrt(3)*sqrt(3)=3?
    # Let's just scatter them a bit further out to avoid initial overlap penalties
    for k in range(5):
        angle = (k + 0.5) * np.pi / 3.0
        # Place them at distance 2.5 * d to be safe
        init_pos.append([2.5 * d * np.cos(angle), 2.5 * d * np.sin(angle)])
    
    init_pos = np.array(init_pos)
    
    # Initial params: positions, S, alpha, theta
    # Estimate S: Max radius of init_pos + 1
    max_r = np.max(np.linalg.norm(init_pos, axis=1))
    S_init = max_r + 1.0
    
    x0 = np.concatenate([init_pos.flatten(), [S_init, 0.0, 0.0]])
    
    best_score = float('inf')
    best_params = x0.copy()
    
    # Multi-start local optimization
    # We have 140s. Each Nelder-Mead run might take a few seconds.
    # Let's do 20 restarts.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Perturb initial solution
        if r == 0:
            x_start = x0
        else:
            # Random restart with larger S to allow expansion
            S_rand = S_init * (1.0 + 0.2 * np.random.rand())
            pos_rand = init_pos + 0.5 * np.random.randn(N, 2)
            x_start = np.concatenate([pos_rand.flatten(), [S_rand, 0.0, 0.0]])
            
        # Use Nelder-Mead
        # maxiter needs to be high, but we have time limit
        # We can't easily interrupt Nelder-Mead mid-iteration without a callback that raises exception.
        # Instead, we can limit maxiter based on time.
        # Or just run it and check time afterwards.
        
        res = minimize(cost_function, x_start, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success and res.fun < best_score:
            # Verify validity strictly?
            # The cost function includes penalties. If cost is close to S, it's good.
            # We should check if the final configuration is actually valid (no overlap, inside).
            params = res.x
            S_final = params[N*2]
            pos_final = params[:N*2].reshape(N, 2)
            alpha_final = params[N*2+1]
            theta_final = params[N*2+2]
            
            # Strict check
            valid = True
            # Check overlap
            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(pos_final[i] - pos_final[j]) < np.sqrt(3) - 1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # Check containment
                base_verts = np.column_stack((np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)))
                c_t = np.cos(theta_final)
                s_t = np.sin(theta_final)
                R = np.array([[c_t, -s_t], [s_t, c_t]])
                verts = base_verts @ R.T
                
                normals = np.column_stack((np.cos(np.deg2rad(alpha_final) + np.arange(6)*np.pi/3), 
                                           np.sin(np.deg2rad(alpha_final) + np.arange(6)*np.pi/3)))
                D = S_final * np.sqrt(3) / 2.0
                
                for i in range(N):
                    v_i = pos_final[i] + verts
                    proj = v_i @ normals.T
                    violations = proj - D
                    if np.max(violations) > 1e-6:
                        valid = False
                        break
                
            if valid:
                best_score = S_final
                best_params = params
        
        # If we found a very good solution (e.g. < 4.0), we can stop early?
        # Theoretical lower bound:
        # Area of 12 hexes = 12 * (3*sqrt(3)/2) = 18*sqrt(3) ≈ 31.18
        # Area of outer hex = 2*sqrt(3)*S^2
        # 2*sqrt(3)*S^2 >= 31.18 => S^2 >= 31.18 / 3.464 = 9.0 => S >= 3.0
        # So 3.0 is a lower bound.
        # If we get < 3.5, it's likely near optimal.
        if best_score < 3.5:
             break

    if best_score == float('inf'):
        # Fallback to a simple valid packing if optimization failed
        # Use the initial cluster with a large S
        S_fallback = 10.0
        inner = [[float(p[0]), float(p[1]), 0.0] for p in init_pos]
        return inner, (0.0, 0.0), S_fallback, 0.0

    # Extract best params
    pos_best = best_params[:N*2].reshape(N, 2)
    S_best = best_params[N*2]
    alpha_best = best_params[N*2+1]
    theta_best = best_params[N*2+2]
    
    inner = []
    for i in range(N):
        inner.append([float(pos_best[i][0]), float(pos_best[i][1]), float(np.rad2deg(theta_best))])
        
    return inner, (0.0, 0.0), float(S_best), float(np.rad2deg(alpha_best))

