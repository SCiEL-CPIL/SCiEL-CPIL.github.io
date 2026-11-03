import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Unit hexagon vertices (pointy-top, angle 0 means vertex at 90 deg? No, let's use standard math angle)
    # Let's define vertices for a hexagon with side 1, centered at origin, angle 0.
    # Angle 0 usually means one vertex at (1,0) or side horizontal.
    # Let's use vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # This corresponds to "pointy-top" if we consider y-axis up?
    # Vertex at 0 deg is (1,0). Vertex at 90 deg is not a vertex.
    # This is a "flat-top" hexagon if we rotate 30?
    # It doesn't matter as long as consistent.
    # Let's use: v_k = (cos(k*pi/3), sin(k*pi/3))
    angles = np.arange(6) * np.pi / 3
    verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Initial State
    # Generate hex grid coordinates for 11 hexagons
    # Center + 6 + 4
    hex_coords = [(0,0)]
    # Ring 1
    hex_coords += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    # Ring 2 - pick 4
    # To make it somewhat symmetric, pick (2,0), (0,2), (-2,1), (-1,-1) ?? 
    # Let's just pick (2,0), (2,-1), (-2,0), (-2,1)
    hex_coords += [(2,0), (2,-1), (-2,0), (-2,1)]
    
    # Convert to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    pos = np.zeros((N, 2))
    for i, (q, r) in enumerate(hex_coords):
        pos[i, 0] = math.sqrt(3) * (q + r/2)
        pos[i, 1] = 1.5 * r
        
    ang = np.zeros(N) # All angle 0
    
    # Objective Function
    def calc_score(p, a):
        # p: (N, 2), a: (N,)
        # Calculate rotation matrices
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        
        # verts: (6, 2)
        # Rotated verts: (N, 6, 2)
        # R = [[cos, -sin], [sin, cos]]
        # v_rot = R * v
        # v_rot_x = cos * v_x - sin * v_y
        # v_rot_y = sin * v_x + cos * v_y
        
        vx = verts[:, 0]
        vy = verts[:, 1]
        
        # Broadcast
        cos_a_b = cos_a[:, np.newaxis]
        sin_a_b = sin_a[:, np.newaxis]
        
        rx = cos_a_b * vx[np.newaxis, :] - sin_a_b * vy[np.newaxis, :]
        ry = sin_a_b * vx[np.newaxis, :] + cos_a_b * vy[np.newaxis, :]
        
        # Add position
        px = p[:, 0:1] + rx
        py = p[:, 1:2] + ry
        
        # Calculate S_req for each vertex
        # S = r * cos((theta mod 60) - 30) / cos(30)
        # theta = atan2(y, x)
        # mod 60: theta % (pi/3)
        # We need to be careful with negative angles.
        # theta_mod = theta % (pi/3)
        # In numpy, % works for floats.
        
        r = np.sqrt(px**2 + py**2)
        theta = np.arctan2(py, px)
        theta_mod = theta % (np.pi / 3)
        
        # Angle offset: 30 deg is pi/6
        # Formula: cos(theta_mod - pi/6)
        # Note: If theta is exactly at a vertex (0, 60...), theta_mod is 0 or 60.
        # cos(0 - 30) = cos(-30) = sqrt(3)/2.
        # r * (sqrt(3)/2) / (sqrt(3)/2) = r. Correct.
        
        s_vals = r * (np.cos(theta_mod - np.pi/6) / (np.sqrt(3)/2))
        
        # Soft max over vertices for each hexagon
        # To avoid overflow, subtract max
        k = 10.0
        s_max_per_hex = np.max(s_vals, axis=1)
        # Soft max: log(sum(exp(k*s))) / k
        # We just want the max, but soft max helps gradients.
        # For simplicity and speed, hard max is often enough for SA.
        # Let's use hard max for speed.
        current_score = np.max(s_vals)
        
        return current_score

    best_score = calc_score(pos, ang)
    best_pos = pos.copy()
    best_ang = ang.copy()
    
    # Simulated Annealing
    T = 1.0
    decay = 0.999
    step_size = 0.5
    
    while time.time() < deadline:
        # Perturb a random hexagon
        i = np.random.randint(N)
        
        new_pos = best_pos.copy()
        new_ang = best_ang.copy()
        
        # Random walk
        new_pos[i, 0] += np.random.normal(0, step_size)
        new_pos[i, 1] += np.random.normal(0, step_size)
        new_ang[i] += np.random.normal(0, step_size * 0.5)
        
        new_score = calc_score(new_pos, new_ang)
        
        # Acceptance
        if new_score < best_score or np.random.rand() < math.exp(-(new_score - best_score) / T):
            best_score = new_score
            best_pos = new_pos
            best_ang = new_ang
            
        T *= decay
        # Reduce step size slowly
        step_size *= 0.9999
        
        # If T is too small, restart with higher T?
        if T < 0.01:
            T = 0.5
            step_size = 0.1

    # Convert back to list of lists
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        a_deg = np.degrees(best_ang[i])
        inner.append([float(x), float(y), float(a_deg)])
        
    return inner, [0.0, 0.0], float(best_score), 0.0
# EVOLVE-BLOCK-END
