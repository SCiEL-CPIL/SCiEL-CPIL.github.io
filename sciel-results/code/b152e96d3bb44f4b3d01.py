import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal Grid
    # We want to place 200 points in a hexagonal pattern.
    # Approximate density: 200 points in area 1.0.
    # Hexagonal packing density is pi/(2*sqrt(3)) ~ 0.9069.
    # Area per point in hex grid ~ 1/sqrt(3)/2 * s^2.
    # 200 * (sqrt(3)/4 * s^2) ~ 1.0  => s^2 ~ 4/(200*sqrt(3)) ~ 0.0115 => s ~ 0.107
    # Let's try s = 0.11
    
    s = 0.11
    hex_points = []
    # Generate a large enough grid to cover the square [-0.5, 0.5]
    # x range: -0.5 to 0.5, y range: -0.5 to 0.5
    # Hex grid:
    # row j: y = j * s * sqrt(3)/2
    # col i: x = i * s + (j % 2) * s/2
    
    max_j = int(1.0 / (s * np.sqrt(3)/2)) + 5
    max_i = int(1.0 / s) + 5
    
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            # Keep points roughly inside the square, with some margin
            if -0.52 <= x <= 0.52 and -0.52 <= y <= 0.52:
                hex_points.append([x, y])
    
    hex_points = np.array(hex_points)
    
    # If we have more than N points, select the N closest to the center (0,0)
    if len(hex_points) > N:
        dists = np.sum(hex_points**2, axis=1)
        idx = np.argsort(dists)
        x0 = hex_points[idx[:N]]
    else:
        # If not enough, pad with random points (unlikely with above params)
        x0 = np.random.rand(N, 2) - 0.5
        # Add the hex points we have
        # This case is unlikely, so we just use random for simplicity if it happens
        x0 = np.random.rand(N, 2) - 0.5

    # Ensure x0 is exactly N points
    x0 = x0.reshape(N, 2)
    
    def compute_min_dist(pts):
        """
        Compute the minimum distance between any two points and the walls.
        This is 2 * radius.
        """
        # Wall distances
        # Distance to wall for point i: min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # Since pts are in [-0.5, 0.5], 0.5 - |x| and 0.5 - |y|
        wall_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        # Use cdist for efficiency? Or manual?
        # N=200, N^2=40000. Manual is fine.
        # To speed up, we can compute only upper triangle
        # But for simplicity and vectorization, let's use a trick or just loop if needed.
        # Actually, scipy.spatial.distance.cdist is fast.
        # But let's stick to numpy for no extra dependency overhead in tight loop.
        
        # Efficient min pairwise distance:
        # We can't easily vectorize min over all pairs without N^2 matrix.
        # 200x200 matrix is small.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_wall, min_pair)

    # 2. Simulated Annealing
    best_x = x0.copy()
    best_score = compute_min_dist(best_x)
    current_x = x0.copy()
    current_score = best_score
    
    # SA Parameters
    T = 0.01  # Initial temperature
    cooling = 0.9995
    min_T = 1e-5
    
    # Move size
    step_size = 0.01
    
    n_iter = 0
    max_iter = 1000000
    
    # Track acceptance rate to adapt step size
    accept_count = 0
    total_count = 0
    
    while time.time() < deadline - 5.0:
        # Generate a move
        # Pick a random point
        i = np.random.randint(N)
        # Gaussian noise
        noise = np.random.normal(0, step_size, 2)
        new_x = current_x.copy()
        new_x[i] += noise
        
        # Keep in bounds (soft or hard?)
        # Hard clipping is better for validity, but might cause issues with gradient.
        # Let's clip to [-0.5, 0.5]
        new_x = np.clip(new_x, -0.5, 0.5)
        
        new_score = compute_min_dist(new_x)
        
        # Acceptance criterion
        if new_score > current_score:
            # Better, always accept
            current_x = new_x
            current_score = new_score
            accept_count += 1
            if current_score > best_score:
                best_score = current_score
                best_x = current_x.copy()
        else:
            # Worse, accept with probability exp((new - old) / T)
            # Note: we are maximizing, so (new - old) is negative
            delta = new_score - current_score
            if delta > -T: # Optimization: if delta is very negative, exp is ~0
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_x = new_x
                    current_score = new_score
                    accept_count += 1
        
        total_count += 1
        
        # Adapt step size every 100 iterations
        if total_count % 100 == 0:
            accept_rate = accept_count / total_count
            if accept_rate > 0.5:
                step_size *= 1.1
            elif accept_rate < 0.1:
                step_size *= 0.9
            # Reset counters
            accept_count = 0
            total_count = 0
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
        
        n_iter += 1

    # 3. Polish with Nelder-Mead
    # Use the best solution found by SA
    x_start = best_x.flatten()
    
    def neg_min_dist(x):
        pts = x.reshape(-1, 2)
        # Penalty for out of bounds
        out_of_bounds = np.max(np.abs(pts) - 0.5)
        if out_of_bounds > 0:
            return 1e6 + out_of_bounds
        return -compute_min_dist(pts)
    
    # Run Nelder-Mead with tight tolerances
    # This helps fine-tune the position
    res = minimize(neg_min_dist, x_start, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.fun < -best_score:
        best_x = res.x.reshape(N, 2)
        best_score = -res.fun
    else:
        # Keep best_x from SA if Nelder-Mead didn't improve
        pass

    # Final check and clip
    best_x = np.clip(best_x, -0.5, 0.5)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_x)):
        return x0 # Fallback to initial grid

    return best_x
