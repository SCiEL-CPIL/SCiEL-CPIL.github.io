"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal lattice is a good starting point for MD
    # We generate a slightly denser hex lattice than necessary to ensure we have >= N points
    # then we will simulate them.
    # Alternatively, random initialization inside the disk.
    # Let's use a hex lattice initialization because it's closer to the optimal structure.
    
    # Estimate spacing for N circles in unit disk. Area of disk = pi. Area per circle ~ pi/r^2.
    # N * pi/r^2 ~ pi => r ~ 1/sqrt(N).
    # For N=300, r ~ 1/17.3 ~ 0.057.
    # Hex spacing s = 2*r.
    
    # Generate a grid of points that covers the unit disk
    # We'll use a simple hexagonal grid generation
    points = []
    # Approximate number of rows/cols needed
    # Let's just generate a lot of points in a hex grid covering [-1.1, 1.1]
    s_init = 0.06 # Initial spacing guess
    step_x = s_init
    step_y = s_init * np.sqrt(3) / 2
    
    x_start = -1.2
    x_end = 1.2
    y_start = -1.2
    y_end = 1.2
    
    i = 0
    x = x_start
    while x <= x_end:
        y = y_start
        offset = 0 if (i % 2 == 0) else step_x / 2
        while y <= y_end:
            points.append([x + offset, y])
            y += step_y
        x += step_x
        i += 1
        
    points = np.array(points)
    
    # Filter points inside the unit circle (with a small margin to avoid boundary issues initially)
    radii = np.linalg.norm(points, axis=1)
    mask = radii <= 1.0
    points = points[mask]
    
    # If we have more than N points, we need to select N.
    # For MD, it's better to start with exactly N points to avoid having to drop them.
    # We can randomly select N points from the lattice points inside the disk.
    # Or, we can start with N random points.
    # Let's stick to the lattice but select the N points closest to the center? No, that leaves the perimeter empty.
    # Better: Start with N points initialized in a hex-like pattern that fits roughly.
    
    # Let's try a different initialization: Random points inside the disk.
    # This avoids bias from the lattice if the lattice doesn't match the optimal packing exactly.
    # However, MD from random can be slow.
    # Let's use the lattice points but if we have too many, we subsample.
    # Actually, the seed program uses a lattice. Let's use a lattice but perturb it slightly to break symmetry and allow MD to find a better local minimum.
    
    # Regenerate lattice specifically to get ~N points
    # Target r ~ 0.053
    # s = 2*r ~ 0.106
    # Let's use s = 0.1
    
    def generate_hex_points(n_target, s):
        pts = []
        # Estimate grid size
        # Area ~ pi. Number of points ~ pi / (s^2 * sqrt(3)/2)
        # Let's just iterate until we have enough
        # Max radius 1.0
        # Rows from -1.0 to 1.0
        n_rows = int(2.0 / (s * np.sqrt(3) / 2)) + 2
        n_cols = int(2.0 / s) + 2
        
        for j in range(-n_rows, n_rows + 1):
            y = j * s * np.sqrt(3) / 2
            if abs(y) > 1.0: continue
            offset = 0 if (j % 2 == 0) else s / 2
            for i in range(-n_cols, n_cols + 1):
                x = i * s + offset
                r = np.hypot(x, y)
                if r <= 1.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) < n_target:
            return None
        # Randomly select n_target points
        # To be more structured, we could select the ones closest to the center, 
        # but random selection from the lattice allows MD to "flow" into better positions.
        # Let's select points uniformly at random from the available lattice points.
        indices = np.random.choice(len(pts), n_target, replace=False)
        return pts[indices]

    # Try to get a good initial set
    # Start with a slightly larger spacing to ensure we can fit them, then MD will compress?
    # No, MD with repulsion will expand them.
    # If we start too close, repulsion is huge.
    # If we start too far, attraction (if any) or just lack of repulsion means they stay far.
    # We want them to push against each other and the wall.
    # So we should start them close together.
    
    # Let's use a spacing that corresponds to r=0.05
    s = 0.1
    pts = generate_hex_points(N, s)
    if pts is None:
        # Fallback to random
        theta = np.random.rand(N) * 2 * np.pi
        r = np.sqrt(np.random.rand(N)) * 0.9
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
    # Add small random noise to break symmetry
    pts = pts + np.random.normal(0, 0.01, size=pts.shape)
    
    # Ensure they are inside the disk initially
    radii = np.linalg.norm(pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        # Project back to boundary
        pts[mask] = pts[mask] / radii[mask, None] * 0.999

    # Molecular Dynamics Simulation
    # We want to maximize the minimum distance.
    # This is equivalent to minimizing the potential energy of repulsive forces.
    # Force F_ij = 1 / d_ij^2 (or similar) if d_ij < 2*r_target, else 0.
    # But we don't know r_target.
    # Standard approach: Minimize the "energy" E = sum_{i<j} (1/d_ij^2) + sum_i (1/d_i^2) where d_i is distance to boundary?
    # No, standard circle packing MD:
    # Use a repulsive force between all pairs.
    # F_ij = k / d_ij^2 (for i != j)
    # F_i_boundary = k / (1 - |c_i|)^2 (if |c_i| > 1 - margin)
    
    # Let's use a simple velocity Verlet integration.
    # Parameters
    dt = 0.01
    k_rep = 1.0
    k_wall = 10.0
    damping = 0.99
    
    velocities = np.zeros_like(pts)
    
    def compute_forces(positions):
        forces = np.zeros_like(positions)
        # Pairwise forces
        # O(N^2) is 300^2 = 90,000 operations. Very fast in numpy.
        # Calculate all pairwise distances
        # To save memory, we can do it in chunks or just full matrix since N=300 is small.
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-10
        dist = np.sqrt(dist_sq)
        
        # Force magnitude: k / d^2
        # We only want repulsion.
        # If we use k/d^2, it's always repulsive.
        # This will try to maximize all distances, which is not what we want.
        # We want to maximize the MINIMUM distance.
        # A common potential for packing is E = sum_{i<j} (1/d_ij^2) which minimizes the sum of inverse squares.
        # This tends to equalize distances.
        # Force F = -dE/dr = 2 * sum_{j != i} 1/d_ij^3 * (r_i - r_j)
        
        # Let's use the force from E = sum 1/d^2
        # F_i = sum_{j != i} 2/d_ij^3 * (p_i - p_j)
        
        inv_dist_cubed = 1.0 / (dist_sq * dist)
        # Set diagonal to 0
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        # Force vector sum
        # F_i = sum_j (2 * inv_dist_cubed_ij * (p_i - p_j))
        # (p_i - p_j) is diff[i, j]
        # So F_i = sum_j (2 * inv_dist_cubed[i, j] * diff[i, j])
        forces_pairwise = 2.0 * np.sum(inv_dist_cubed[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces
        # If a point is too close to the wall (r > 1 - eps), push it in.
        # Let's define a "soft" wall.
        # Force towards center if r > 1.0? No, they must be inside.
        # If r > 1.0, push in strongly.
        # If r < 1.0, no force from wall?
        # Actually, we want them to push against the wall to maximize the packing density.
        # So we need an attractive force to the wall? Or just let them move freely and constrain?
        # If we only have repulsion between circles, they will fly apart to infinity.
        # We need a confining potential.
        # Let's use a potential that penalizes being outside the disk (r > 1) and also penalizes being too far from the boundary?
        # No, the standard "maximize min distance" is tricky with just MD.
        
        # Alternative: Use a potential that penalizes pairs closer than a target distance R.
        # But we don't know R.
        
        # Let's use the "Thomson problem" style potential: E = sum 1/d^2.
        # This minimizes the energy by spreading points out.
        # To keep them in the disk, we add a wall potential.
        # Wall potential: If r > 1, E_wall = k_wall * (r - 1)^2.
        # Force: -dE/dr = -2 * k_wall * (r - 1) * (r_vec / r).
        # This pushes them back if they go outside.
        # But will they push against the wall?
        # If the repulsion between circles is strong, they will push each other and the outer ones will be pushed against the wall.
        
        # Let's implement this.
        
        r_vec = positions
        r_mags = np.linalg.norm(r_vec, axis=1)
        r_mags_safe = np.where(r_mags > 1e-10, r_mags, 1e-10)
        
        forces_wall = np.zeros_like(positions)
        # Only apply if r > 1.0
        outside_mask = r_mags > 1.0
        if np.any(outside_mask):
            # Force = -2 * k_wall * (r - 1) * (r_vec / r)
            # Note: gradient of (r-1)^2 w.r.t r_vec is 2(r-1) * r_vec/r
            # So force is -2 * k_wall * (r-1) * r_vec/r
            forces_wall[outside_mask] = -2.0 * k_wall * (r_mags[outside_mask] - 1.0)[:, np.newaxis] * (r_vec[outside_mask] / r_mags[outside_mask, np.newaxis])
            
        return forces_pairwise + forces_wall

    # Run MD
    steps = 1000
    for step in range(steps):
        if time.time() > deadline - 10:
            break
            
        forces = compute_forces(pts)
        
        # Update velocity
        velocities = velocities * damping + forces * dt
        # Update position
        pts = pts + velocities * dt
        
        # Hard constraint: Keep inside disk
        # If any point goes outside, project it back
        r_mags = np.linalg.norm(pts, axis=1)
        outside_mask = r_mags > 1.0
        if np.any(outside_mask):
            pts[outside_mask] = pts[outside_mask] / r_mags[outside_mask, np.newaxis]
            # Zero out velocity component normal to boundary? Or just damp it.
            velocities[outside_mask] *= 0.5

    # After MD, we have a configuration.
    # Now refine using local optimization.
    # We can try to maximize the minimum distance by optimizing the positions of the most constrained points.
    # Or use scipy.optimize.minimize on the whole set with a penalty function.
    
    # Let's define the objective function to minimize:
    # We want to maximize min_dist.
    # Equivalent to minimizing -min_dist.
    # But -min_dist is not differentiable.
    # We can minimize a smooth approximation: sum(1/d_ij^2) for pairs close to the min distance?
    # Or just use the MD result and check the score.
    
    # Let's try a simple local search:
    # Identify the pair with the minimum distance.
    # Try to move them apart slightly.
    
    # Calculate current min distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    # Also check boundary distance
    r_mags = np.linalg.norm(pts, axis=1)
    min_boundary_dist = 1.0 - np.max(r_mags)
    
    current_radius = min(min_dist / 2.0, min_boundary_dist)
    
    # If we have time, try to improve using scipy
    if time.time() < deadline - 5:
        # Define objective for scipy
        # We want to maximize the minimum distance.
        # Let's minimize the "energy" of the closest pairs.
        # Find the k closest pairs.
        flat_dist_sq = dist_sq.flatten()
        flat_dist_sq = flat_dist_sq[flat_dist_sq < np.inf]
        k_closest = 10
        if len(flat_dist_sq) >= k_closest:
            # Get indices of k closest
            idx = np.argpartition(flat_dist_sq, k_closest)[:k_closest]
            closest_pairs = []
            for i in idx:
                r, c = np.unravel_index(i, dist_sq.shape)
                if r != c:
                    closest_pairs.append((r, c))
            
            # We can try to optimize the positions of the points involved in these pairs.
            # This is complex.
            # Simpler: Use a penalty function that penalizes any distance less than a target R.
            # Start with R = current_radius.
            # Increase R slightly and see if we can find a configuration.
            
            # Let's try a simple hill climbing on the whole configuration.
            # Move random points by a small amount if it increases the min distance.
            
            best_pts = pts.copy()
            best_radius = current_radius
            step_size = 0.01
            iterations = 1000
            
            for _ in range(iterations):
                if time.time() > deadline - 2:
                    break
                
                # Pick a random point
                idx = np.random.randint(N)
                # Try a random move
                delta = np.random.normal(0, step_size, size=2)
                new_pts = best_pts.copy()
                new_pts[idx] += delta
                
                # Check if still inside
                new_r = np.linalg.norm(new_pts[idx])
                if new_r > 1.0:
                    continue
                
                # Calculate new min distance
                # Only need to check distances involving idx
                # d_ij for j != idx
                diffs = new_pts[idx] - new_pts
                diffs[idx] = 1e10 # Exclude self
                dists_sq = np.sum(diffs**2, axis=1)
                min_pair_dist_sq = np.min(dists_sq)
                min_pair_dist = np.sqrt(min_pair_dist_sq)
                
                # Boundary dist for idx
                min_boundary = 1.0 - new_r
                
                # Other points' boundary distances didn't change
                # But other points' pair distances didn't change?
                # Yes, only distances involving idx changed.
                # So the global min distance is min(old_min_dist_not_involving_idx, new_min_dist_involving_idx, min_boundary)
                
                # We need the old min distance not involving idx.
                # This is expensive to calculate every time.
                # Let's just recalculate the global min distance. N=300 is small enough.
                
                diff_all = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
                dist_sq_all = np.sum(diff_all**2, axis=-1)
                np.fill_diagonal(dist_sq_all, np.inf)
                global_min_dist = np.sqrt(np.min(dist_sq_all))
                
                r_mags_all = np.linalg.norm(new_pts, axis=1)
                global_min_boundary = 1.0 - np.max(r_mags_all)
                
                new_radius = min(global_min_dist / 2.0, global_min_boundary)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_pts = new_pts
                else:
                    # Decay step size if we haven't improved in a while?
                    # For now, just keep trying
                    pass
                    
            pts = best_pts
            current_radius = best_radius

    return pts
# EVOLVE-BLOCK-END
