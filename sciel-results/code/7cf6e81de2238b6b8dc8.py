"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(points):
    if len(points) == 0:
        return 0.0
    # min distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # min distance between points
    # To avoid O(N^2) memory explosion for large N in a simple loop, 
    # we use a chunked approach or just accept O(N^2) time for N=300 (90k pairs)
    # 300 points is small enough for O(N^2) in Python if vectorized partially, 
    # but full NxN matrix is 90k floats, which is tiny.
    x = points[:, 0]
    y = points[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_boundary, min_pair / 2.0)
    return r if r > 0 else 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: Hexagonal lattice
    # We generate a dense hex lattice and pick N points
    # Or better, start with a random distribution in the disk to avoid local minima of lattice?
    # Lattice is a good start.
    
    def generate_hex_lattice(n_points):
        # Estimate spacing
        # Area = pi, n circles area = n * pi * r^2
        # r ~ sqrt(1/n)
        # spacing ~ 2r
        r_est = 1.0 / np.sqrt(n_points)
        spacing = 2.0 * r_est
        
        # Generate points in a bounding box
        # Radius of bounding box needs to be 1
        # Number of rows/cols
        side = 2.0
        k = int(side / spacing) + 2
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) == 0:
            return np.random.uniform(-0.9, 0.9, (n_points, 2))
        
        # If we have more than n_points, we need to select or shrink
        # If we have less, we expand?
        # For N=300, spacing 2/sqrt(300) ~ 0.115.
        # Hex lattice in unit circle usually fits ~ 100-200 points with that spacing?
        # Let's just generate a lot and take the first N closest to center?
        # Or better: generate a grid of candidates and pick N.
        
        # Actually, for annealing, it's better to start with exactly N points.
        # If lattice has > N, take random N? No, take central N.
        # If lattice has < N, we need to generate more.
        
        # Let's try a different initialization: Random uniform in disk.
        # Random uniform in disk is often better for annealing to escape lattice traps.
        angles = np.random.uniform(0, 2*np.pi, n_points)
        radii = np.sqrt(np.random.uniform(0, 1, n_points))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    points = generate_hex_lattice(N)
    
    # Ensure initial validity (no overlaps, inside circle)
    # If random, they might overlap. 
    # For annealing, we can handle overlaps with high energy penalty.
    
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    # Simulated Annealing
    # Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Force calculation
    # We want to maximize r = min(min_pair/2, min_boundary)
    # This is equivalent to minimizing the maximum "violation" or maximizing the minimum distance.
    # A common approach: 
    # 1. Calculate current r.
    # 2. Identify "critical" pairs (those defining the min distance) and boundary violations.
    # 3. Apply forces to increase these distances.
    
    # Simpler approach: 
    # Objective function to minimize: max(0, 2r_target - dist) ? No, we don't know r_target.
    # Let's use a potential energy function:
    # E = sum( max(0, 1 - dist_ij / (2*r_est))^2 ) + sum( max(0, |p_i| - (1-r_est))^2 )
    # This is tricky because r_est changes.
    
    # Alternative: Directly move points apart.
    # For each pair (i,j), if dist < 2*r_current, push them apart.
    # But r_current is global.
    
    # Let's use a standard "expand" heuristic:
    # 1. Compute all pair distances and boundary distances.
    # 2. Find the minimum distance d_min and boundary violation b_min.
    # 3. If d_min/2 < b_min, the limiting factor is pairs.
    #    Apply repulsive force to the closest pair (or all pairs below a threshold).
    # 4. If b_min < d_min/2, the limiting factor is boundary.
    #    Push points towards center.
    # 5. Add random noise scaled by T.
    
    # To be efficient, we can just do random moves and accept if radius improves (hill climbing) 
    # or with some probability (annealing).
    # Hill climbing is fast but gets stuck. Annealing is slower but escapes.
    # Given 110s, we can do many iterations.
    
    # Let's implement a "soft" force-directed method.
    # Force on i from j: F = k / d^2 * (dir) if d < 2*r_target else 0.
    # But r_target is unknown.
    
    # Let's stick to: 
    # 1. Compute r.
    # 2. Identify set of "tight" constraints (dist < 2*r + eps, or boundary < r + eps).
    # 3. Move points to relieve these constraints.
    
    # Even simpler: Random perturbation + Accept if better (with annealing probability).
    
    T = 0.1
    step_size = 0.05
    
    while time.time() < deadline and T > T_min:
        # Compute current radius
        r_curr = _compute_radius(points)
        
        # If we improved, update best
        if r_curr > best_radius:
            best_radius = r_curr
            best_points = points.copy()
        
        # Generate a perturbation
        # Random move
        delta = np.random.normal(0, step_size, (N, 2))
        new_points = points + delta
        
        # Keep inside circle (hard constraint for validity, but we can allow slight violation in search?)
        # The evaluator requires valid output. 
        # If we allow invalid intermediate states, we must ensure final state is valid.
        # Easier: Project back to valid region?
        # If a point goes outside, push it back to boundary.
        dists = np.linalg.norm(new_points, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / dists[mask][:, None] * 0.999
        
        # Compute new radius
        r_new = _compute_radius(new_points)
        
        # Acceptance criterion
        if r_new >= r_curr:
            points = new_points
        else:
            # Metropolis criterion
            # Energy difference: (r_curr - r_new) is the "cost"
            # Probability: exp(-cost / T)
            # Cost is in units of radius. T is in units of radius.
            delta_r = r_curr - r_new
            if delta_r < T:
                if np.random.rand() < np.exp(-delta_r / T):
                    points = new_points
        
        # Cooling
        T *= cooling_rate
        
        # Adjust step size?
        # If we are stuck (no improvement in last 100 steps), increase T or step size?
        # Simple: keep T schedule.
        
    # Final check: ensure best_points is valid
    # The evaluator checks: min_pair/2 >= r and min_boundary >= r
    # Our _compute_radius returns the max valid r for the configuration.
    # So best_points is valid for radius best_radius.
    
    return best_points
# EVOLVE-BLOCK-END
