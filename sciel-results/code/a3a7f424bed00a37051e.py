"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialize with a hexagonal lattice
    # We need to generate enough points and pick the best N
    # A hexagonal lattice with spacing s has points at (i*s, j*s*sqrt(3)/2) with offset
    # We want to fill a circle of radius 1.
    
    def init_hex():
        best_points = None
        best_count = 0
        # Spacing determines density. Smaller spacing = more points.
        # We want roughly N points in area pi.
        # Area per point in hex lattice ~ s^2 * sqrt(3)/2.
        # N * s^2 * sqrt(3)/2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
        s_est = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            s = s_est * scale
            # Generate points in a square grid large enough to cover unit circle
            # Max radius is 1. 
            # x in [-1, 1], y in [-1, 1]
            # Number of points in x: ~ 2/s. Let's use a generous bound.
            n_pts = int(1.0 / s) + 3
            points = []
            for j in range(-n_pts, n_pts + 1):
                for i in range(-n_pts, n_pts + 1):
                    x = i * s
                    y = j * s * np.sqrt(3) / 2
                    if j % 2 != 0:
                        x += s / 2
                    r = np.hypot(x, y)
                    if r <= 1.0:
                        points.append((x, y))
            
            if len(points) >= N:
                # Sort by distance from center to pick the most central N
                # This ensures a symmetric, compact start
                points_arr = np.array(points)
                dists = np.linalg.norm(points_arr, axis=1)
                idx = np.argsort(dists)
                points = points_arr[idx[:N]]
                
                # Check if this configuration is valid (no overlaps? hex lattice has fixed spacing)
                # The radius of circles would be s/2 if they touch.
                # We just need a good start.
                if best_points is None or len(points) > best_count:
                    best_points = points
                    best_count = len(points)
                    
        if best_points is None:
             # Fallback: random
             return np.random.rand(N, 2) * 2 - 1
        return best_points

    points = init_hex()
    
    # 2. Bottleneck Gradient Ascent Optimization
    
    def get_radius_and_grads(pts):
        # Returns r, grad (N, 2)
        # r = min(min_pair_dist/2, min_boundary_dist)
        
        # Boundary distances
        norms = np.linalg.norm(pts, axis=1)
        b_dist = 1.0 - norms
        
        # Pairwise distances
        # N=50, so 50x50 matrix is fine
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_boundary_dist = np.min(b_dist)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        grads = np.zeros_like(pts)
        
        # Determine which constraint is active (bottleneck)
        # We sum gradients from all active constraints (within a small epsilon) to handle ties
        
        eps = 1e-6
        
        # 1. Boundary constraints
        active_boundary = np.where(b_dist <= min_boundary_dist + eps)[0]
        for k in active_boundary:
            if norms[k] > 1e-9:
                # We want to increase 1 - |c_k|, so decrease |c_k|
                # Gradient of (1 - |c_k|) wrt c_k is -c_k/|c_k|
                # We want to move in direction of gradient to increase value
                # So add -c_k/|c_k| to grads[k]
                grads[k] -= pts[k] / norms[k]
            else:
                # If at center, any direction increases distance? No, distance is 0.
                # Move randomly or stay? If r is limited by boundary at center, it's 1.0.
                # Unlikely to be the bottleneck if we have 50 circles.
                pass

        # 2. Pairwise constraints
        # Find all pairs with dist <= min_pair_dist + eps
        active_pairs = np.argwhere(dists <= min_pair_dist + eps)
        # Ensure we don't process (i,j) and (j,i) both, and i!=j
        # argwhere returns all indices.
        # We can just iterate and if i<j process.
        for i, j in active_pairs:
            if i >= j: continue
            d = dists[i, j]
            if d < 1e-9: continue
            vec = pts[j] - pts[i]
            unit_vec = vec / d
            
            # We want to increase d.
            # Gradient of d wrt c_i is -unit_vec
            # Gradient of d wrt c_j is +unit_vec
            # We want to maximize d, so we move in direction of gradient.
            # So grads[i] += -unit_vec
            # grads[j] += +unit_vec
            # Note: r = d/2. Gradient wrt r is same direction.
            grads[i] -= unit_vec
            grads[j] += unit_vec
            
        return r, grads

    current_r, _ = get_radius_and_grads(points)
    best_points = points.copy()
    best_r = current_r
    
    step_size = 0.01
    max_iters = 10000
    no_improve_count = 0
    
    for i in range(max_iters):
        r, grads = get_radius_and_grads(points)
        
        # Normalize gradient to avoid huge steps if multiple constraints
        grad_norm = np.linalg.norm(grads, axis=1)
        # Avoid division by zero
        grad_norm[grad_norm < 1e-9] = 1.0
        normalized_grads = grads / grad_norm[:, np.newaxis]
        
        # Update
        new_points = points + normalized_grads * step_size
        
        # Check validity: must be inside unit circle
        # If a point goes outside, it's invalid. 
        # We can project it back or reject the step.
        # Projection:
        norms = np.linalg.norm(new_points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            # If any point is outside, this step is invalid for the "inside" constraint
            # But wait, the objective includes boundary distance.
            # If a point goes outside, boundary distance is negative.
            # The function get_radius would return a negative r?
            # min(1 - |c|) would be negative.
            # We want to maximize r. Negative r is bad.
            # So we should avoid going outside.
            # Let's project points that are outside back to the boundary?
            # Or just reject the step?
            # Rejecting is safer.
            # But if we are stuck at the boundary, we might need to move along it.
            # The gradient for boundary constraint pushes IN.
            # So if we are at the boundary, the gradient points inward.
            # So we shouldn't go out.
            # If we do, it's due to pairwise repulsion.
            # In that case, the pairwise force is stronger.
            # We should probably reduce step size.
            
            # For now, let's project to be safe and continue
            new_points[outside] = new_points[outside] / norms[outside, np.newaxis] * 0.999
            
        new_r, _ = get_radius_and_grads(new_points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Increase step size slightly if improving consistently?
            # step_size = min(step_size * 1.05, 0.1)
        else:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-5:
                # Try random restart or jitter
                # Add small random noise to escape local minima
                points = best_points + np.random.normal(0, 0.01, (N, 2))
                # Project back inside
                norms = np.linalg.norm(points, axis=1)
                outside = norms > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / norms[outside, np.newaxis] * 0.999
                step_size = 0.01
                no_improve_count = 0
                if i > max_iters - 100:
                    break
                    
    return best_points
# EVOLVE-BLOCK-END
