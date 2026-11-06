import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal Lattice
    def generate_hex_lattice(d):
        if d <= 0:
            return np.array([])
        
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure circles are inside: dist + r <= 1 => dist <= 1 - d/2
                if dist + d/2.0 <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        if len(points) >= N:
            # Select the N points closest to the center
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # Find a good initial lattice spacing d
    # r = d/2. We want to maximize r, so minimize d.
    # We need at least N points.
    d_low = 0.05  # Likely yields > 300 points
    d_high = 0.15 # Likely yields < 300 points
    
    # Ensure d_low is valid (enough points)
    while True:
        pts = generate_hex_lattice(d_low)
        if len(pts) >= N:
            break
        d_low *= 1.1
        
    # Ensure d_high is invalid (not enough points)
    while True:
        pts = generate_hex_lattice(d_high)
        if len(pts) < N:
            break
        d_high *= 0.9

    # Binary search for the smallest d that yields >= N points
    for _ in range(30):
        d_mid = (d_low + d_high) / 2.0
        pts = generate_hex_lattice(d_mid)
        if len(pts) >= N:
            d_low = d_mid
        else:
            d_high = d_mid
            
    points = generate_hex_lattice(d_low)
    
    if len(points) < N:
        # Fallback: Random uniform in disk, scaled to fit
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Scale down to ensure they fit with some margin
        max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
        if max_r > 0:
            points *= (0.9 / max_r)

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]

    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.001, (N, 2))

    # Ensure initial validity
    init_radii = np.hypot(points[:, 0], points[:, 1])
    outside = init_radii > 1.0
    if np.any(outside):
        points[outside] = points[outside] / init_radii[outside, np.newaxis] * 0.999

    best_points = points.copy()
    
    def compute_radius(pts):
        """
        Compute the maximum radius r such that all circles are disjoint and inside the unit circle.
        r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
        """
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Boundary constraint
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        if min_boundary <= 0:
            return 0.0

        # Pairwise distance constraint
        # For N=300, a 300x300 matrix is 90,000 elements. This is fast in NumPy.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Set diagonal to inf to ignore self-distances
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_sq = np.min(dists_sq)
        min_pair = np.sqrt(min_pair_sq) / 2.0
        
        return min(min_pair, min_boundary)

    def compute_radius_fast(pts):
        """
        Fast approximation of the minimum radius using a spatial grid.
        """
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Boundary constraint
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        if min_boundary <= 0:
            return 0.0

        # Estimate minimum pairwise distance
        # Use a grid cell size slightly larger than the expected minimum distance
        # Since we don't know the exact min distance, we use a heuristic.
        # If we assume r ~ 0.05, min dist ~ 0.1. Cell size 0.1 is good.
        cell_size = 0.1 
        min_dist = float('inf')
        
        # Grid dimensions
        x_min, y_min = -1.1, -1.1
        x_max, y_max = 1.1, 1.1
        grid_w = int((x_max - x_min) / cell_size)
        grid_h = int((y_max - y_min) / cell_size)
        
        grid = {}
        
        for i in range(N):
            x, y = pts[i]
            gx = int((x - x_min) / cell_size)
            gy = int((y - y_min) / cell_size)
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if i < j: # Avoid double counting
                                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                                if d < min_dist:
                                    min_dist = d
                                    if min_dist < 2 * min_boundary:
                                        # Early exit if boundary is tighter
                                        pass 
                                      
            if (gx, gy) not in grid:
                grid[(gx, gy)] = []
            grid[(gx, gy)].append(i)

        min_pair = min_dist / 2.0
        if min_dist == float('inf'):
             min_pair = min_boundary
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # Optimization Loop
    # Strategy: Target-aware force relaxation using spatial grid for forces.
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 30 # Reduced to allow more frequent perturbations
    
    # Force scaling parameters
    lambda_pair = 1.0
    lambda_bound = 1.0
    
    # Grid parameters for force calculation
    # We need a cell size such that only nearby points are checked.
    # If target radius is r, min dist is 2r. Cell size ~ 2r is sufficient.
    # We will update cell size based on best_r.
    
    for it in range(50000):
        if time.time() > deadline:
            break
            
        # Compute current radius periodically
        # Use fast check first
        current_r_fast = compute_radius_fast(points)
        
        if current_r_fast > best_r * 0.99: # Potential improvement
             # Verify with exact calculation
             current_r = compute_radius(points)
             if current_r > best_r:
                 best_r = current_r
                 best_points = points.copy()
                 no_improve_count = 0
             else:
                 no_improve_count += 1
        else:
             no_improve_count += 1
             
        if no_improve_count > max_no_improve:
            # Perturbation
            radii = np.hypot(points[:, 0], points[:, 1])
            boundary_proximity = 1.0 - radii
            min_bound_val = np.min(boundary_proximity)
            close_to_boundary = boundary_proximity < (min_bound_val + 0.005 * best_r)
            
            num_perturb = 10
            if np.sum(close_to_boundary) > 0:
                idxs = np.where(close_to_boundary)[0]
                chosen = np.random.choice(idxs, size=min(num_perturb, len(idxs)), replace=False)
                points[chosen] += np.random.normal(0, 0.005 * best_r, (len(chosen), 2))
            
            random_idx = np.random.choice(N, size=num_perturb, replace=False)
            points[random_idx] += np.random.normal(0, 0.003 * best_r, (num_perturb, 2))
            
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
                
            no_improve_count = 0
            step_size *= 0.9

        # Compute Forces using Grid
        r_target = best_r * 1.005
        forces = np.zeros_like(points)
        
        # Grid setup for forces
        # Cell size should be large enough to capture interactions.
        # Interactions matter if dist < 2*r_target.
        # So cell_size = 2*r_target is a good choice.
        cell_size = 2.0 * r_target
        if cell_size < 0.01: cell_size = 0.01
        
        x_min, y_min = -1.1, -1.1
        x_max, y_max = 1.1, 1.1
        grid_w = int((x_max - x_min) / cell_size)
        grid_h = int((y_max - y_min) / cell_size)
        
        grid = {}
        
        # Assign points to grid
        for i in range(N):
            x, y = points[i]
            gx = int((x - x_min) / cell_size)
            gy = int((y - y_min) / cell_size)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        # Calculate pair forces
        # We iterate over points and check neighbors
        # To avoid double counting or to ensure Newton's 3rd law, we can just sum forces on i from j.
        # Since force is symmetric and anti-parallel, we can just compute force on i from all j in neighbor cells.
        
        for i in range(N):
            x_i, y_i = points[i]
            gx = int((x_i - x_min) / cell_size)
            gy = int((y_i - y_min) / cell_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if i != j:
                                x_j, y_j = points[j]
                                dx_p = x_i - x_j
                                dy_p = y_i - y_j
                                d_sq = dx_p**2 + dy_p**2
                                d = np.sqrt(d_sq)
                                if d < 2.0 * r_target:
                                    if d < 1e-6:
                                        # Random push if coincident
                                        forces[i, 0] += np.random.uniform(-1, 1) * 0.1
                                        forces[i, 1] += np.random.uniform(-1, 1) * 0.1
                                    else:
                                        violation = 2.0 * r_target - d
                                        if violation > 0:
                                            # F = k * violation / d^3 * (p_i - p_j)
                                            # direction = (dx_p, dy_p) / d
                                            # magnitude = violation / d^2
                                            # F = (dx_p, dy_p) * (violation / d^3)
                                            factor = violation / (d**3)
                                            forces[i, 0] += dx_p * factor
                                            forces[i, 1] += dy_p * factor
                                            
        # Boundary Forces
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_violation = r_target - (1.0 - radii)
        active_boundary = boundary_violation > 0
        
        if np.any(active_boundary):
            r_safe = np.where(radii > 1e-6, radii, 1e-6)
            norm_vec = -points / r_safe[:, np.newaxis]
            bound_factors = boundary_violation / r_target
            bound_factors[~active_boundary] = 0.0
            bound_forces = bound_factors[:, np.newaxis] * norm_vec
            forces += lambda_bound * bound_forces

        # Update positions
        points += forces * step_size
        
        # Enforce hard constraint
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
            
        # Decay step size
        step_size *= 0.9999

    # Final Polish Phase
    if time.time() < deadline - 5.0:
        points = best_points.copy()
        polish_deadline = deadline - 2.0
        polish_step = 0.001
        
        for it in range(20000):
            if time.time() > polish_deadline:
                break
            
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
            
            r_target = best_r * 1.001
            
            forces = np.zeros_like(points)
            
            # Use grid for polish forces too
            cell_size = 2.0 * r_target
            if cell_size < 0.01: cell_size = 0.01
            
            x_min, y_min = -1.1, -1.1
            x_max, y_max = 1.1, 1.1
            grid = {}
            
            for i in range(N):
                x, y = points[i]
                gx = int((x - x_min) / cell_size)
                gy = int((y - y_min) / cell_size)
                key = (gx, gy)
                if key not in grid:
                    grid[key] = []
                grid[key].append(i)
                
            for i in range(N):
                x_i, y_i = points[i]
                gx = int((x_i - x_min) / cell_size)
                gy = int((y_i - y_min) / cell_size)
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        key = (gx + dx, gy + dy)
                        if key in grid:
                            for j in grid[key]:
                                if i != j:
                                    x_j, y_j = points[j]
                                    dx_p = x_i - x_j
                                    dy_p = y_i - y_j
                                    d_sq = dx_p**2 + dy_p**2
                                    d = np.sqrt(d_sq)
                                    if d < 2.0 * r_target:
                                        if d < 1e-6:
                                            forces[i, 0] += np.random.uniform(-1, 1) * 0.01
                                            forces[i, 1] += np.random.uniform(-1, 1) * 0.01
                                        else:
                                            violation = 2.0 * r_target - d
                                            if violation > 0:
                                                factor = violation / (d**3)
                                                forces[i, 0] += dx_p * factor
                                                forces[i, 1] += dy_p * factor
                                                
            radii = np.hypot(points[:, 0], points[:, 1])
            boundary_violation = r_target - (1.0 - radii)
            active_boundary = boundary_violation > 0
            if np.any(active_boundary):
                r_safe = np.where(radii > 1e-6, radii, 1e-6)
                norm_vec = -points / r_safe[:, np.newaxis]
                bound_factors = boundary_violation / r_target
                bound_factors[~active_boundary] = 0.0
                bound_forces = bound_factors[:, np.newaxis] * norm_vec
                forces += lambda_bound * bound_forces

            points += forces * polish_step
            
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
                
            polish_step *= 0.9999

    return best_points
