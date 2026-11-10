"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_init(n):
    """
    Generate initial points on a hexagonal lattice within [-0.5, 0.5]^2.
    Hexagonal packing is denser than square packing, providing a better start.
    """
    # Hexagonal lattice spacing for N points approx sqrt(2*N/sqrt(3))
    # Area of unit square is 1. Density of hex packing is pi/(2*sqrt(3)) ~ 0.9069
    # Area per circle ~ N / (1 / 0.9069) = N * 0.9069
    # Radius r ~ sqrt(Area_per_circle / pi)
    # Spacing d = 2r ~ sqrt(4 * Area_per_circle / pi) = sqrt(4 * N * 0.9069 / pi)
    
    # Let's just generate a grid and filter to fit N points, or generate enough and truncate.
    # Better: Generate a large hex grid, center it, and pick the N closest to center?
    # Or just generate rows.
    
    # Spacing between rows in hex lattice: dy = sqrt(3)/2 * dx
    # Let dx be the horizontal spacing.
    # We want to pack ~ N points.
    # Approximate dx:
    # N ~ (1/dx) * (1/(sqrt(3)/2 * dx)) * 2 (accounting for 2D area roughly)
    # Actually, density of points per area in hex lattice with spacing dx is 2/(sqrt(3)*dx^2).
    # We want N points in area 1.
    # N = 2 / (sqrt(3) * dx^2)  => dx = sqrt(2 / (sqrt(3) * N))
    
    dx = np.sqrt(2.0 / (np.sqrt(3.0) * N))
    dy = dx * np.sqrt(3.0) / 2.0
    
    # Generate grid
    # x from -0.5 to 0.5
    # y from -0.5 to 0.5
    # Offset odd rows by dx/2
    
    points = []
    y = -0.5
    row_idx = 0
    while y <= 0.5:
        x_start = -0.5
        if row_idx % 2 == 1:
            x_start = -0.5 + dx/2.0
            
        x = x_start
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            x += dx
        y += dy
        row_idx += 1
        
    points = np.array(points)
    
    if len(points) < N:
        # If we didn't generate enough (shouldn't happen with this dx), pad with random
        # Or just resize dx to be smaller
        return points[:N] if len(points) >= N else np.random.rand(N, 2) - 0.5
        
    # If we have more than N points, we should pick the N that are "best".
    # Usually, we want the ones closest to the center to maximize boundary distance?
    # Or just truncate? Truncating might leave holes.
    # Better: Sort by distance to center and take the N closest?
    # Actually, for packing, we want a contiguous block.
    # Let's just take the first N from a centered generation.
    
    # Re-generate centered
    points = []
    # Estimate number of rows needed
    num_rows = int(np.ceil(1.0 / dy)) + 2
    num_cols = int(np.ceil(1.0 / dx)) + 2
    
    y_start = -0.5
    for r in range(num_rows):
        y = y_start + r * dy
        if y > 0.5: break
        x_start = -0.5
        if r % 2 == 1:
            x_start = -0.5 + dx/2.0
            
        for c in range(num_cols):
            x = x_start + c * dx
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < N:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # If we have more, we need to reduce. 
    # A simple heuristic: remove points from the corners/edges that are furthest from the "bulk"?
    # Or just take the N closest to the origin? This creates a hexagonal cluster in the center.
    # This is actually a very good starting point for maximizing the minimum distance to the boundary 
    # and between points, as it avoids the "wasted" space at corners initially.
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:N]]

def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    d = pdist(pts, 'euclidean')
    min_pair = d.min() if len(d) > 0 else 0.0
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill

    # 1. Hexagonal Initialization
    pts = _hex_init(N)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    np.random.seed(42)
    
    # 2. Vectorized Gradient-Free Optimization
    # We use a soft-min surrogate for the objective to allow gradient-like steps 
    # or large parallel updates.
    # Objective: maximize r = min( min_pair_dist/2, min_boundary_dist )
    # Surrogate: f(pts) = - (logsumexp(-beta * constraints) / beta)
    # where constraints are distances.
    # To maximize min(d_i), we minimize -min(d_i).
    # Soft min: -logsumexp(beta * d_i) / beta.
    # As beta -> inf, this approaches -min(d_i).
    
    # We will use a simple stochastic gradient ascent on the soft objective.
    # Since we don't have easy analytic gradients for the global min, we use 
    # a finite difference approximation or a "shake and bake" approach.
    # However, a better "different design" is using a **repulsive force model** 
    # with adaptive temperature (Simulated Annealing style) but vectorized.
    
    # Let's stick to a robust vectorized local search with adaptive steps.
    # The key improvement over the seed is:
    # 1. Hex init (better start)
    # 2. Vectorized distance calculation for multiple candidates at once (faster)
    # 3. Adaptive step size based on improvement rate
    
    current_pts = best_pts.copy()
    current_r = best_r
    
    # Pre-compute indices for pair distances? 
    # pdist is fast. 
    # To speed up local search, we can compute the "bottleneck" pairs and only check those?
    # No, moving a point can change the global min.
    
    # Strategy: 
    # 1. Coarse phase: Large steps, random moves.
    # 2. Fine phase: Small steps, targeted moves.
    
    # We will implement a "Parallel Trial" local search.
    # For each iteration, pick K points to move.
    # Generate M candidates for each.
    # Evaluate all K*M candidates in batch (if possible) or sequentially but fast.
    
    # Given the 120s limit, we need to be efficient.
    # pdist on 200 points is very fast (~microseconds).
    # The bottleneck in the seed was likely the Python loop overhead and random direction generation.
    
    # Let's use a simpler, faster loop with NumPy vectorization for the *candidates*.
    
    phase_configs = [
        {'steps': 500, 'step_size': 0.05, 'trials': 5},
        {'steps': 500, 'step_size': 0.02, 'trials': 10},
        {'steps': 1000, 'step_size': 0.005, 'trials': 20},
        {'steps': 2000, 'step_size': 0.001, 'trials': 50},
    ]
    
    for config in phase_configs:
        if time.time() > deadline:
            break
            
        steps = config['steps']
        base_step = config['step_size']
        n_trials = config['trials']
        
        for it in range(steps):
            if time.time() > deadline:
                break
            
            # Pick a random point
            idx = np.random.randint(N)
            orig = current_pts[idx].copy()
            
            # Generate n_trials random candidates
            # Directions
            dirs = np.random.randn(n_trials, 2)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / norms
            
            # Steps with variation
            steps_var = base_step * (0.5 + np.random.rand(n_trials))
            
            # New positions
            candidates = orig[None, :] + dirs * steps_var[:, None]
            candidates = np.clip(candidates, -0.499, 0.499)
            
            best_move_r = current_r
            best_move = None
            
            # Evaluate candidates
            # We can't easily batch pdist for different points in the same array easily without
            # expanding the array. 
            # For N=200, a single pdist is ~ O(N^2) = 40000 ops.
            # Doing this n_trials times is 5-50 * 40000 = 200k - 2M ops.
            # This is fast in C (scipy).
            
            for k in range(n_trials):
                temp_pts = current_pts.copy()
                temp_pts[idx] = candidates[k]
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > best_move_r:
                    best_move_r = temp_r
                    best_move = candidates[k]
            
            if best_move is not None:
                current_pts[idx] = best_move
                current_r = best_move_r
                
                if current_r > best_r:
                    best_r = current_r
                    best_pts = current_pts.copy()
                    
            # Adaptive step size: if no improvement, shrink step?
            # Or if we hit deadline, break.
            
    # Final polish
    if time.time() < deadline:
        current_pts = best_pts.copy()
        current_r = best_r
        
        # Very fine steps
        for it in range(1000):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            orig = current_pts[idx].copy()
            
            # 100 tiny trials
            dirs = np.random.randn(100, 2)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / norms
            steps_var = 0.0001 * (0.5 + np.random.rand(100))
            candidates = orig[None, :] + dirs * steps_var[:, None]
            candidates = np.clip(candidates, -0.499, 0.499)
            
            best_move_r = current_r
            best_move = None
            
            for k in range(100):
                temp_pts = current_pts.copy()
                temp_pts[idx] = candidates[k]
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > best_move_r:
                    best_move_r = temp_r
                    best_move = candidates[k]
            
            if best_move is not None:
                current_pts[idx] = best_move
                current_r = best_move_r
                
                if current_r > best_r:
                    best_r = current_r
                    best_pts = current_pts.copy()

    best_pts = np.clip(best_pts, -0.499, 0.499)
    return best_pts
# EVOLVE-BLOCK-END
