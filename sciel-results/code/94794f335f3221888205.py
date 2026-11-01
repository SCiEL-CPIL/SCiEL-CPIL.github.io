import numpy as np

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between circles and to the boundary.
    Uses Simulated Annealing with a penalty-based energy function.
    """
    
    # Target radius: The best known packing radius.
    # We aim to find a configuration where all circles of this radius fit.
    r_target = 0.071377103865
    
    # 1. Initialization
    # Start with a jittered grid to provide a reasonable starting point.
    # A 7x7 grid has 49 points, we need 50.
    # Let's use a 10x5 grid or similar, then jitter.
    # Actually, a random uniform distribution is often sufficient for SA to find the global minimum if the landscape is explored well,
    # but a grid helps avoid initial high energy states.
    
    # Create a 10x5 grid (50 points)
    n_x, n_y = 10, 5
    x_coords = np.linspace(-0.45, 0.45, n_x)
    y_coords = np.linspace(-0.45, 0.45, n_y)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
    # Add small random jitter to break symmetry
    np.random.seed(42) # For reproducibility in testing, though SA is stochastic
    points += np.random.uniform(-0.01, 0.01, points.shape)
    
    # Ensure points are strictly inside the square initially
    points = np.clip(points, -0.49, 0.49)

    def calculate_energy(pts, r):
        """
        Calculate the energy penalty for a configuration of points.
        Energy is the sum of squared overlaps with other circles and the boundary.
        """
        # Pairwise distances
        # Using cdist for efficiency
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Overlaps with other circles
        # Penalty for each pair where dist < r
        overlaps = np.maximum(0, r - dists)
        pair_energy = np.sum(overlaps ** 2)
        
        # Boundary violations
        # Distance to each wall
        d_left = 0.5 + pts[:, 0]
        d_right = 0.5 - pts[:, 0]
        d_down = 0.5 + pts[:, 1]
        d_up = 0.5 - pts[:, 1]
        
        # Take min distance to any wall for each point
        # Actually, we need to check if the circle extends beyond the wall.
        # dist_to_wall_i = min(d_left_i, d_right_i, d_down_i, d_up_i)
        # If dist_to_wall_i < r, there is a violation.
        min_dist_to_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        
        wall_overlaps = np.maximum(0, r - min_dist_to_wall)
        wall_energy = np.sum(wall_overlaps ** 2)
        
        return pair_energy + wall_energy

    # 2. Simulated Annealing
    current_energy = calculate_energy(points, r_target)
    best_energy = current_energy
    best_points = points.copy()
    
    # Temperature schedule
    T_start = 0.01
    T_end = 1e-8
    n_steps = 200000 # Number of SA steps
    
    # Pre-calculate indices for faster updates? 
    # For N=50, cdist is fast enough (50x50 matrix).
    
    # We will update one point at a time.
    # To speed up energy calculation, we can compute the change in energy for a single point move.
    # Delta E = (New Pair Energy for point i) - (Old Pair Energy for point i) + (New Wall Energy for point i) - (Old Wall Energy for point i)
    
    def get_point_energy_contrib(pts, i, r):
        """
        Calculate the energy contribution of point i to the total energy.
        This includes its interaction with all other points and the walls.
        Note: Pair energy is sum over i<j. If we move i, we affect terms (i,j) for all j!=i.
        """
        # Wall energy for i
        d_left = 0.5 + pts[i, 0]
        d_right = 0.5 - pts[i, 0]
        d_down = 0.5 + pts[i, 1]
        d_up = 0.5 - pts[i, 1]
        min_dist_to_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        wall_overlap = np.maximum(0, r - min_dist_to_wall)
        wall_e = wall_overlap ** 2
        
        # Pair energy for i
        # Distances to all other points
        diffs = pts - pts[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        # Exclude self (distance 0)
        dists[i] = np.inf
        pair_overlaps = np.maximum(0, r - dists)
        pair_e = np.sum(pair_overlaps ** 2)
        
        return wall_e + pair_e

    current_energies = np.array([get_point_energy_contrib(points, i, r_target) for i in range(N)])
    # Note: The sum of individual contributions is not exactly the total energy because pair energy is double counted 
    # in the sum of individual contributions? 
    # Total Energy E = sum_{i<j} (r-d_ij)^2 + sum_i (r-d_i_wall)^2
    # Sum of individual contributions S = sum_i [ sum_{j!=i} (r-d_ij)^2 + (r-d_i_wall)^2 ]
    # S = 2 * sum_{i<j} (r-d_ij)^2 + sum_i (r-d_i_wall)^2
    # So E = (S - sum_i (r-d_i_wall)^2) / 2 + sum_i (r-d_i_wall)^2 = S/2 + 0.5 * sum_i (r-d_i_wall)^2
    # This is getting complicated to track deltas accurately without recalculating.
    # Given N=50, recalculating full energy is O(N^2) = 2500 ops. 200,000 steps * 2500 = 500M ops.
    # In Python, this might be too slow.
    # Let's optimize: Use numpy vectorization for the full energy calculation, but maybe reduce steps or optimize the inner loop.
    # Or, implement the delta calculation correctly.
    
    # Let's try to optimize the delta calculation.
    # E_total = E_pairs + E_walls
    # E_pairs = 0.5 * sum_i sum_{j!=i} max(0, r - d_ij)^2
    # E_walls = sum_i max(0, r - d_i_wall)^2
    
    # If we move point i:
    # Delta E_walls = New_Wall_E(i) - Old_Wall_E(i)
    # Delta E_pairs = 0.5 * (New_Pair_Sum(i) - Old_Pair_Sum(i))
    # Where Pair_Sum(i) = sum_{j!=i} max(0, r - d_ij)^2
    
    # We need to store the current "Pair_Sum" for each i? 
    # Or just recalculate the pair part for i.
    
    # Let's stick to full recalculation but use a more efficient Python loop or reduce steps.
    # 200k steps might be too many for full recalc in pure Python if not optimized.
    # Let's try 50,000 steps first and see. Or use a faster energy calc.
    
    # Alternative: Use a library like scipy.optimize.minimize with method='COBYLA' or 'SLSQP' on the min-dist function?
    # No, SA is better for non-convex.
    
    # Let's refine the SA to be faster.
    # We can compute the energy incrementally.
    
    # Precompute current distances? No, they change.
    
    # Let's try a hybrid: Run SA for a few thousand steps with full recalc, then switch to local search?
    # Or just use a very efficient energy calculation.
    
    # Let's try to make the full energy calculation as fast as possible using NumPy.
    def fast_energy(pts, r):
        # Pairwise
        # Subtract each row from each column
        # This creates a N x N x 2 array, which is 50*50*2 = 5000 floats. Very small.
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        overlaps = np.maximum(0, r - dists)
        pair_e = np.sum(overlaps**2)
        
        # Walls
        d_left = 0.5 + pts[:, 0]
        d_right = 0.5 - pts[:, 0]
        d_down = 0.5 + pts[:, 1]
        d_up = 0.5 - pts[:, 1]
        
        min_dist_to_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        wall_overlaps = np.maximum(0, r - min_dist_to_wall)
        wall_e = np.sum(wall_overlaps**2)
        
        return pair_e + wall_e

    # 50k steps of fast_energy (which is vectorized numpy) should be fast enough.
    # Each fast_energy call is ~microseconds to milliseconds.
    # 50k * 1ms = 50s. Might be tight.
    # Let's try 20k steps.
    
    n_steps = 20000
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate a random move
        # Move size should decrease with temperature? Or stay constant?
        # Let's use a move size proportional to sqrt(T) or fixed small value.
        move_scale = 0.01 * np.sqrt(T / T_start) + 0.001
        move = np.random.normal(0, move_scale, 2)
        
        new_point = points[i].copy()
        new_point += move
        
        # Keep inside square
        new_point = np.clip(new_point, -0.49, 0.49)
        
        # Calculate delta energy
        # To save time, we only recalculate energy if the move is potentially promising?
        # No, we need Delta E for acceptance.
        
        old_energy = current_energy
        points[i] = new_point
        new_energy = fast_energy(points, r_target)
        delta_e = new_energy - old_energy
        
        # Acceptance
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            points[i] = points[i] - move # Revert
            # Note: points[i] was modified in place. We need to revert correctly.
            # Better: keep a copy of the old point.
            
        # Wait, the revert logic above is flawed if move was clipped.
        # Let's do it properly:
        
    # Re-doing the loop with proper state management
    
    current_energy = fast_energy(points, r_target)
    best_energy = current_energy
    best_points = points.copy()
    
    for step in range(n_steps):
        i = np.random.randint(N)
        old_point = points[i].copy()
        
        move_scale = 0.01 * np.sqrt(T / T_start) + 0.001
        move = np.random.normal(0, move_scale, 2)
        
        new_point = old_point + move
        new_point = np.clip(new_point, -0.49, 0.49)
        
        points[i] = new_point
        new_energy = fast_energy(points, r_target)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            points[i] = old_point
            
        T *= cooling_rate

    # After SA, we might have a very low energy, but maybe not zero.
    # If energy is not zero, the circles overlap.
    # We should return the best points found.
    # The score is determined by the actual min distance of the returned points.
    
    return best_points

# EVOLVE-BLOCK-END
