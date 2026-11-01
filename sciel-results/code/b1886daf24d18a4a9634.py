import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 120.0  # Keep a safety margin before 150s

    # Define 12 points in a honeycomb lattice (axial coordinates q, r)
    # We use a 3x4 rhombus patch
    q_vals = [0, 1, 2]
    r_vals = [0, 1, 2, 3]
    
    points = []
    for r in r_vals:
        for q in q_vals:
            # Convert axial to cartesian
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            points.append([x, y])
    
    points = np.array(points)
    
    # Center the points
    center = np.mean(points, axis=0)
    points -= center

    def get_min_hex_side(points_rotated):
        # Calculate the minimum side length L' of a hexagon (angle 0) containing points_rotated
        # Constraints:
        # |x| <= L'
        # |y| <= sqrt(3)/2 * L'
        # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * L'
        # |sqrt(3)/2 * x - 1/2 * y| <= sqrt(3)/2 * L'
        
        x = points_rotated[:, 0]
        y = points_rotated[:, 1]
        
        # L' must satisfy:
        # L' >= max(|x|)
        # L' >= max(|y|) / (sqrt(3)/2)
        # L' >= max(|sqrt(3)/2 * x + 1/2 * y|) / (sqrt(3)/2)
        # L' >= max(|sqrt(3)/2 * x - 1/2 * y|) / (sqrt(3)/2)
        
        L1 = np.max(np.abs(x))
        L2 = np.max(np.abs(y)) / (np.sqrt(3) / 2.0)
        L3 = np.max(np.abs((np.sqrt(3) / 2.0) * x + 0.5 * y)) / (np.sqrt(3) / 2.0)
        L4 = np.max(np.abs((np.sqrt(3) / 2.0) * x - 0.5 * y)) / (np.sqrt(3) / 2.0)
        
        return max(L1, L2, L3, L4)

    def objective(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rotation matrix
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        points_rot = (points @ R.T)
        
        L_prime = get_min_hex_side(points_rot)
        
        # Outer side length S
        S = np.sqrt(3) * L_prime + 1.0
        
        return S

    # Ternary search for optimal theta in [0, pi/3]
    # The problem is symmetric every pi/3
    a = 0.0
    b = np.pi / 3.0
    
    # Perform ternary search
    # Number of iterations to ensure precision
    # Each iteration takes ~1ms. 100 iterations is fast.
    for _ in range(100):
        if time.time() > deadline:
            break
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        
        f1 = objective(m1)
        f2 = objective(m2)
        
        if f1 < f2:
            b = m2
        else:
            a = m1
            
    theta_opt = (a + b) / 2.0
    
    # Calculate final points
    cos_t = np.cos(theta_opt)
    sin_t = np.sin(theta_opt)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    final_points = (points @ R.T)
    
    # Construct inner hexagons
    # Each inner hexagon is at final_points[i], angle 0
    inner = []
    for i in range(N):
        x, y = final_points[i]
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length S
    S = objective(theta_opt)
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
