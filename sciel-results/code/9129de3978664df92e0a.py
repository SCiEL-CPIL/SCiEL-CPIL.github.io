import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    # 1. Initialization: Hexagonal lattice
    # Generate a hexagonal grid
    # Ideal spacing for 200 points in unit square
    # Area = 1, N = 200
    # Area per point = 1/200 = 0.005
    # Hex area per point = d^2 * sqrt(3)/2
    # d = sqrt(0.005 * 2 / sqrt(3))
    d = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate points
    # We need enough points to cover the square
    # Square side 1. Hex width approx d.
    # Number of points in x direction ~ 1/d
    # Let's generate a grid from -0.5 to 0.5
    # x = i * d
    # y = j * d * sqrt(3)/2 + (i % 2) * d * sqrt(3)/4
    
    max_idx = int(np.ceil(0.5 / d)) + 2
    xs = []
    ys = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            x = i * d
            y = j * d * np.sqrt(3) / 2 + (i % 2) * d * np.sqrt(3) / 4
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                xs.append(x)
                ys.append(y)
    
    points = np.array([xs, ys]).T
    
    # If we have more than N points, we need to select N
    # If we have fewer, we might have scaled too big or small.
    # For d calculated above, we should have approx N points.
    # Let's check count.
    if len(points) > N:
        # Select points closest to the center? 
        # Or select a subset that maintains the lattice structure?
        # Simple heuristic: Sort by distance from center, take the first N?
        # This creates a circular cluster.
        # Better: Randomly sample? No.
        # Let's just take the first N if they are well distributed.
        # Actually, the generation loop above produces a square-ish patch of hex lattice.
        # If count > N, we can just take a random subset? 
        # Or better, we can scale down d slightly to get fewer points?
        # Let's try to adjust d if count is off.
        # But for now, let's just take the first N if > N, or all if < N.
        # To be safe, if > N, we can remove points that are furthest from the origin?
        # No, that leaves a hole in the center? No, closest to origin are in the center.
        # So taking closest to origin leaves the center filled and edges empty.
        # This is good for a square? Not really, we want to fill the square.
        
        # Alternative: Just use the points we have. If > N, take a random permutation and slice N?
        # This breaks the lattice but provides a valid start for optimization.
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # This shouldn't happen with the d estimate, but if it does, we need to add points.
        # Or we can just resize the array? No, N is fixed.
        # Let's assume d estimate is close enough.
        # If we have fewer, we can't construct N points.
        # We should probably generate more candidates.
        # Let's regenerate with a smaller d to ensure we have enough.
        pass # Handled below if needed
    
    # Ensure we have exactly N points
    if len(points) < N:
        # Fallback: Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
    else:
        points = points[:N]

    # 2. Optimization
    # We will use a simple gradient-free "push" method.
    # Maximize r = min( min_{i<j} dist(i,j)/2, min_i dist(i, boundary) )
    
    # Precompute boundary constraints
    # Boundary distance for point i:
    # min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
    
    # We iterate:
    # 1. Calculate all pairwise distances and boundary distances.
    # 2. Find the global minimum radius r.
    # 3. For each point, calculate the direction to move to increase its local minimum clearance.
    #    Local clearance for i is min( min_j dist(i,j), min_k dist(i, boundary_k) )
    #    Note: The global score is min( global_min_pair_dist / 2, global_min_boundary_dist ).
    #    To improve the score, we need to increase the bottleneck.
    #    The bottleneck is either a pair (i,j) or a point-boundary (i,b).
    #    
    #    Strategy:
    #    For each point i, find the "closest" neighbor j and "closest" boundary b.
    #    Let d_j = dist(i,j) and d_b = dist(i,b).
    #    The local constraint for i is min(d_j, d_b).
    #    However, moving i affects j.
    #    Simple approach: Move each point in the direction of the vector sum of unit vectors pointing away from its closest neighbors and closest boundary.
    #    Weight the movement by the "slack" or just use a fixed step.
    
    step_size = 0.01
    decay = 0.999
    
    # Use a list of points for easy updating? No, numpy array is better.
    # We need to be careful with simultaneous updates.
    # Gauss-Seidel style (update one by one) is often better for convergence but slower in Python.
    # Jacobi style (update all based on old positions) is faster but can oscillate.
    # Let's use Jacobi with a small step.
    
    num_iterations = 5000
    for it in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Calculate distances
        # Pairwise distances
        # dist[i,j] = ||c_i - c_j||
        # We only need the min for each i.
        
        # Efficient way to find min neighbor distance for each point:
        # This is O(N^2).
        # Let's compute the full distance matrix? 200x200 = 40k floats. Fast.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_mat, np.inf)
        min_neighbor_dist = np.min(dist_mat, axis=1)
        
        # Boundary distances
        # d_b = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
        d_b = np.min([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ], axis=0)
        
        # Current radius estimate
        # r = min( min(min_neighbor_dist)/2, min(d_b) )
        current_r = min(np.min(min_neighbor_dist) / 2.0, np.min(d_b))
        
        # Calculate forces
        # For each point i, we want to move away from the constraint that is limiting it.
        # If min_neighbor_dist[i] < d_b[i], the neighbor is the bottleneck.
        # Else, the boundary is the bottleneck.
        
        forces = np.zeros_like(points)
        
        for i in range(N):
            # Find closest neighbor
            j = np.argmin(dist_mat[i])
            
            if min_neighbor_dist[i] < d_b[i]:
                # Move away from neighbor j
                vec = points[i] - points[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    forces[i] += vec / norm
            else:
                # Move away from boundary
                # Determine which boundary is closest
                # d_b components:
                db_x_neg = 0.5 - points[i, 0]
                db_x_pos = 0.5 + points[i, 0]
                db_y_neg = 0.5 - points[i, 1]
                db_y_pos = 0.5 + points[i, 1]
                
                # Find min
                min_val = np.min([db_x_neg, db_x_pos, db_y_neg, db_y_pos])
                
                # Add force from all boundaries that are close (within some epsilon of min)
                # Or just the closest one.
                if db_x_neg == min_val:
                    forces[i, 0] -= 1.0
                if db_x_pos == min_val:
                    forces[i, 0] += 1.0
                if db_y_neg == min_val:
                    forces[i, 1] -= 1.0
                if db_y_pos == min_val:
                    forces[i, 1] += 1.0
            
            # Also, if the other constraint is very close, we might want to consider it?
            # For simplicity, just the active constraint.
            
        # Update positions
        # Normalize forces?
        # The magnitude of force is 1.
        # We want to move in the direction of the force.
        # But if multiple points push each other, we need to balance.
        # Simple update: points += step_size * forces
        # But we must ensure we don't cross boundaries or overlap?
        # The step size is small, so it should be okay.
        
        # Apply update
        points += step_size * forces
        
        # Project back to square?
        # If a point goes outside, the boundary force should have pushed it in.
        # But numerical errors might push it out.
        # Clamp to square with a small margin?
        # No, clamping breaks the smooth optimization.
        # The force logic should keep them in.
        # However, if a point is at the boundary, force is inward.
        
        # Decay step size
        step_size *= decay
        
    # Final cleanup: ensure validity
    # The optimization might have resulted in invalid states if step size was too large initially.
    # Let's do a final check and repair if necessary.
    # If any point is outside, move it inside.
    # If any pair overlaps, push them apart.
    
    # Repair pass
    for _ in range(10):
        # Check boundaries
        for i in range(N):
            x, y = points[i]
            if x < -0.5: points[i, 0] = -0.5
            if x > 0.5: points[i, 0] = 0.5
            if y < -0.5: points[i, 1] = -0.5
            if y > 0.5: points[i, 1] = 0.5
        
        # Check overlaps
        # Simple pairwise check
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(points[i] - points[j])
                # We don't know the target radius exactly, but we want them disjoint.
                # The score is determined by the min dist.
                # If they are too close, push them apart.
                # What is "too close"?
                # Let's just push them apart if dist < 0.01 (arbitrary small value)?
                # No, we want to maximize the min dist.
                # The optimization should have handled this.
                # If the optimization failed, we might have overlaps.
                # Let's just ensure they are not coincident.
                if dist < 1e-9:
                    points[j] += 0.001
                    changed = True
    
    return points

# EVOLVE-BLOCK-END
