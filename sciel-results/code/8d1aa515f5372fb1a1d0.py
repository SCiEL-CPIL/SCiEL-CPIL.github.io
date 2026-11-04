import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit to be safe
    
    # Define candidate configurations in axial coordinates (i, j)
    # Convert to cartesian: x = sqrt(3)*(i + j/2), y = 1.5*j
    
    configs = []
    
    # Config 1: 9-block (i,j in [-1,1]) + (2,0) + (-2,0)
    c1 = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            c1.append((i, j))
    c1.append((2, 0))
    c1.append((-2, 0))
    configs.append(c1)
    
    # Config 2: 7-cluster + (2,0) + (-2,0) + (0,2) + (0,-2)
    c2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    c2.extend([(2,0), (-2,0), (0,2), (0,-2)])
    configs.append(c2)
    
    # Config 3: 1-3-3-3-1 shape
    c3 = []
    c3.append((0, 2))
    for j in [1, 0, -1]:
        for i in [-1, 0, 1]:
            c3.append((i, j))
    c3.append((0, -2))
    configs.append(c3)
    
    # Config 4: A more compact 11-hex cluster
    # Try 1-2-3-3-2? (11)
    c4 = []
    c4.append((0, 2))
    for i in [-1, 0]: c4.append((i, 1))
    for i in [-1, 0, 1]: c4.append((i, 0))
    for i in [-1, 0, 1]: c4.append((i, -1))
    for i in [0, 1]: c4.append((i, -2))
    configs.append(c4)
    
    best_score = float('inf')
    best_config = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    best_side = 0.0
    
    # Precompute normals for outer hexagon (flat-top, side horizontal)
    # Normals at 90, 150, 210, 270, 330, 30 degrees
    angles_deg = [90, 150, 210, 270, 330, 30]
    normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles_deg])
    
    for config in configs:
        if time.time() > deadline:
            break
            
        # Convert axial to cartesian
        points = []
        for i, j in config:
            x = math.sqrt(3) * (i + j/2.0)
            y = 1.5 * j
            points.append([x, y])
        points = np.array(points)
        
        # Center the points
        center = np.mean(points, axis=0)
        points_centered = points - center
        
        # Optimize rotation angle
        # Sample angles from 0 to 60 degrees
        angles = np.linspace(0, 60, 121)
        min_side_for_config = float('inf')
        best_angle_for_config = 0.0
        
        for theta_deg in angles:
            if time.time() > deadline:
                break
            theta = math.radians(theta_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated_points = points_centered @ R.T
            
            # Calculate max projection onto normals
            # proj = rotated_points @ normals.T
            # max_proj = max over all points and all normals
            # We need the max absolute value? No, the hexagon is centered.
            # The distance to side is d. The point must be within d in all normal directions.
            # So max dot product must be <= d.
            # Since the hexagon is symmetric, we check max dot product for each normal.
            # Actually, we need max over all points of max over all normals of dot(p, n).
            # But the hexagon contains the origin, so we need |dot(p, n)| <= d?
            # No, the hexagon is convex and contains origin.
            # The condition is dot(p, n_k) <= d for all k.
            # And dot(p, -n_k) <= d is implied if we have all 6 normals.
            # So we just take the max dot product over all points and all 6 normals.
            
            projections = rotated_points @ normals.T
            max_proj = np.max(projections)
            
            # L * sqrt(3)/2 = max_proj
            # L = 2 * max_proj / sqrt(3)
            side = 2.0 * max_proj / math.sqrt(3)
            
            if side < min_side_for_config:
                min_side_for_config = side
                best_angle_for_config = theta_deg
        
        if min_side_for_config < best_score:
            best_score = min_side_for_config
            best_config = config
            best_angle = best_angle_for_config
            best_center = (center[0], center[1])
            best_side = best_score
            
    # If no config found, return a fallback
    if best_config is None:
        # Fallback: linear packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Construct the final inner list
    # We need to return (x, y, angle_degrees) for each inner hexagon.
    # The angle of the inner hexagons should match the global rotation?
    # The task says "each inner hexagon may translate and rotate".
    # In our model, all inner hexagons are aligned with the lattice.
    # If we rotate the whole cluster by best_angle, the inner hexagons are also rotated.
    # So their angle is best_angle.
    
    inner = []
    for i, j in best_config:
        x = math.sqrt(3) * (i + j/2.0)
        y = 1.5 * j
        # Apply rotation
        theta = math.radians(best_angle)
        x_rot = x * math.cos(theta) - y * math.sin(theta)
        y_rot = x * math.sin(theta) + y * math.cos(theta)
        # Translate to center
        x_final = x_rot + best_center[0]
        y_final = y_rot + best_center[1]
        inner.append([x_final, y_final, best_angle])
        
    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
