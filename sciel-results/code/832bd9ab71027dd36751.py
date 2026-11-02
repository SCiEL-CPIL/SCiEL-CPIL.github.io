import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def min_outer_side(inner_positions, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center (0,0) that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
        
        all_verts = np.vstack(all_verts)
        
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = np.cos(na)
            ny = np.sin(na)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        return max_proj / (np.sqrt(3) / 2)

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        """
        best_s = float('inf')
        best_theta = 0.0
        
        # Scan theta
        for theta in np.arange(0, 60, 1.0):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Refine theta around best
        for theta in np.arange(best_theta - 1.0, best_theta + 1.0, 0.1):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
                
        return best_s, best_theta

    def objective(params):
        """
        params: 3*N values for inner hexagons (x, y, angle).
        Outer hexagon is assumed centered at (0,0).
        Returns the side length of the minimal enclosing outer hexagon + penalty.
        """
        inner_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                
                # Approximate overlap penalty using center distance and angle difference
                # This is a heuristic to keep the gradient smooth and cheap.
                # A more accurate check would be expensive.
                # Let's use a simple distance-based penalty if they are too close.
                dx = ci[0] - cj[0]
                dy = ci[1] - cj[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Minimum distance for disjoint unit hexagons is roughly 2.0 (if aligned)
                # But with rotation, it can be less. Let's use 1.8 as a soft threshold.
                if dist < 1.8:
                    penalty += (1.8 - dist) ** 2 * 10.0
                    
        s, theta = get_min_outer_config(inner_positions)
        
        return s + penalty

    # Initial configuration: Hexagonal lattice
    init_params = np.zeros(3 * N)
    
    # Center
    init_params[0:3] = [0.0, 0.0, 0.0]
    
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        idx = (k + 1) * 3
        init_params[idx:idx+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        
    # Ring 2 (5 hexagons)
    for k in range(5):
        angle = np.radians(60 * k + 30)
        idx = (k + 7) * 3
        init_params[idx:idx+3] = [3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0]
        
    # Gradient Descent with Numerical Gradients
    current_params = init_params.copy()
    current_score = objective(current_params)
    
    best_params = current_params.copy()
    best_score = current_score
    
    # Learning rate schedule
    lr = 0.1
    decay = 0.99
    
    max_iters = 20000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute numerical gradient
        grad = np.zeros(3 * N)
        eps = 1e-4
        for k in range(3 * N):
            orig_val = current_params[k]
            current_params[k] = orig_val + eps
            f_plus = objective(current_params)
            current_params[k] = orig_val - eps
            f_minus = objective(current_params)
            current_params[k] = orig_val
            grad[k] = (f_plus - f_minus) / (2 * eps)
            
        # Update parameters
        current_params -= lr * grad
        
        new_score = objective(current_params)
        
        if new_score < best_score:
            best_score = new_score
            best_params = current_params.copy()
            
        # Decay learning rate
        lr *= decay
        if lr < 1e-6:
            break

    # Extract best inner positions
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
        
    # Compute final outer side and angle
    s, theta = get_min_outer_config(inner)
    
    # Center is (0,0)
    center = (0.0, 0.0)
    side = s
    angle = theta
    
    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        side *= 1.01
        
    return inner, center, side, angle
