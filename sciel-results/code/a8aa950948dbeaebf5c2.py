import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Hexagonal Lattice
    # This creates a dense, regular packing which is a good starting point.
    # We generate points in a hex grid and keep those inside the unit circle.
    # We might need to adjust spacing to get close to N points, or just generate
    # enough and then pick the best N or optimize all N.
    # Actually, for fixed N, we should generate a grid with roughly the right density
    # and then use the first N points that fit, or better, just generate a grid
    # and take N points, then optimize.
    
    # Estimate spacing: Area = pi * 1^2. N=300. Area per circle ~ pi/300.
    # Hex packing density ~ 0.9069.
    # r_circle ~ sqrt( (pi/300) * 0.9069 / pi ) ? No.
    # Area of hexagon for side s is (3*sqrt(3)/2) * s^2.
    # Distance between centers d = 2*r.
    # Let's just create a grid with d=0.1 and see how many points we get.
    
    d = 0.1
    x = np.arange(-1.2, 1.2, d)
    y = np.arange(-1.2, 1.2, d)
    X, Y = np.meshgrid(x, y)
    
    # Hexagonal offset for every other row
    for i in range(len(Y)):
        if i % 2 == 1:
            Y[i, :] += d / 2.0
            
    points_all = np.column_stack((X.ravel(), Y.ravel()))
    
    # Keep only points inside unit circle
    norms = np.linalg.norm(points_all, axis=1)
    mask = norms < 1.0
    points_init = points_all[mask]
    
    # We need exactly N points.
    # If we have more, we can randomly select N.
    # If we have fewer, we need to add random points or increase density.
    # With d=0.1, area ~ 2.4*2.4=5.76, points ~ 576. Inside circle ~ 450.
    # So we have enough.
    
    if len(points_init) >= N:
        # Randomly select N points to avoid bias
        indices = np.random.choice(len(points_init), N, replace=False)
        points = points_init[indices].copy()
    else:
        # Fallback: Add random points
        points = points_init.copy()
        while len(points) < N:
            r = np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            points = np.vstack([points, [r * np.cos(theta), r * np.sin(theta)]])

    # Ensure points are strictly inside to avoid initial invalidity
    # Project to circle if needed
    norms = np.linalg.norm(points, axis=1)
    outside = norms >= 1.0
    if np.any(outside):
        points[outside] *= 0.99 / norms[outside]

    # 2. Optimization: Simulated Annealing
    # We maximize the minimum radius r.
    # E = -r. We want to minimize E.
    
    def calc_min_radius(pts):
        # Distance to boundary
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        
        # Distance between points
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    current_r = calc_min_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # SA Parameters
    T = 0.05  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 1e-5
    max_iter = 20000  # Number of steps
    no_improve_count = 0
    max_no_improve = 500
    step_scale = 0.05  # Scale for perturbation
    
    import time
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    for i in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        # Select a random point to move
        idx = np.random.randint(N)
        
        # Generate a perturbation
        # The step size should be related to temperature
        delta = np.random.normal(0, T * step_scale, 2)
        
        old_pos = points[idx].copy()
        new_pos = old_pos + delta
        
        # Check if new position is inside circle. If not, reject immediately or project?
        # Rejecting is standard for hard constraints in SA, but projecting can help.
        # Let's reject if it goes outside, to maintain validity.
        if np.linalg.norm(new_pos) >= 1.0:
            continue
            
        points[idx] = new_pos
        
        new_r = calc_min_radius(points)
        delta_E = current_r - new_r  # We maximize r, so improvement is new_r > current_r
        # Energy change for minimization of -r: E_new - E_old = -new_r - (-current_r) = current_r - new_r
        # If delta_E < 0, it's an improvement.
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            points[idx] = old_pos
            no_improve_count += 1
            
        # Cool down
        T *= alpha
        
        if no_improve_count > max_no_improve:
            # Restart or kick
            # If stuck, try a larger kick
            T = max(T, 0.01)
            no_improve_count = 0
            if T < T_min:
                break

    return best_points
# EVOLVE-BLOCK-END
