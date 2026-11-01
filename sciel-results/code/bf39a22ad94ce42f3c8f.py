import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for overhead

    def generate_hex_grid(n_target, max_radius=1.0):
        # Estimate spacing
        # Area of hex cell ~ s^2 * sqrt(3)/2
        # Total Area ~ pi * max_radius^2
        # n_target * s^2 * sqrt(3)/2 ~ pi
        s = np.sqrt(2 * np.pi / (n_target * np.sqrt(3)))
        
        # Generate grid
        k = int(max_radius / s) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= max_radius*max_radius:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n_target, keep the central ones
        if len(points) > n_target:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind='stable')
            points = points[indices[:n_target]]
        elif len(points) < n_target:
            # Try smaller spacing
            s *= 0.95
            return generate_hex_grid(n_target, max_radius)
            
        return points

    # 1. Initialize
    # Try a few different initializations and pick the best one after a short warm-up?
    # Or just use hex. Hex is usually very good for disk packing.
    points = generate_hex_grid(N, 0.99) # Slightly smaller to ensure boundary margin
    
    # Ensure we have exactly N points
    if len(points) < N:
        # Fallback: Random
        points = np.random.uniform(-0.9, 0.9, (N, 2))
    elif len(points) > N:
        points = points[:N]

    # 2. Optimization: Simulated Annealing
    # Objective: Maximize r = min( min_{i<j} ||c_i - c_j||, min_i (1 - ||c_i||) ) / 2
    
    def compute_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        r_vec = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary_dist = np.min(1.0 - r_vec)
        
        return min(min_pair_dist, min_boundary_dist) / 2.0

    current_r = compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    # SA Parameters
    T = 0.001  # Initial temperature (scale of r change)
    alpha = 0.999  # Cooling factor
    max_iters = 20000
    
    # We will use a random walk
    # Step size for perturbation
    step_scale = 0.01
    
    for iter in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Generate perturbation
        # Move all points by a small random amount
        # Or move a subset? Moving all is simpler and often works well for dense packing.
        noise = np.random.normal(0, step_scale, points.shape)
        new_points = points + noise
        
        # Ensure new points are within the container (hard constraint)
        # If a point goes outside, we can either reject the move or project it.
        # Projecting might cause overlaps. Rejecting is safer for SA.
        r_vec_new = np.hypot(new_points[:, 0], new_points[:, 1])
        if np.any(r_vec_new > 1.0):
            continue
            
        new_r = compute_radius(new_points)
        
        # Acceptance criterion
        if new_r > current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Metropolis criterion
            delta = current_r - new_r
            if delta < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                points = new_points
                current_r = new_r
        
        # Cool down
        T *= alpha
        
        # If T is too small, we might be stuck.
        # We can reset T if no improvement for a while.
        if iter % 1000 == 0:
             # Check if we are stuck
             if best_r < current_r + 1e-6: # Not strictly stuck, just checking
                 pass
                 
    # 3. Polish: Local refinement
    # Use a gradient-like method on the best solution found
    # Maximize min distance.
    # This is equivalent to pushing the closest pairs apart.
    
    def polish(pts, time_limit=10.0):
        start = time.time()
        pts = pts.copy()
        step_size = 0.005
        
        for i in range(5000):
            if time.time() - start > time_limit:
                break
                
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_pair_dist = np.min(dists)
            r_vec = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary_dist = np.min(1.0 - r_vec)
            
            current_r = min(min_pair_dist, min_boundary_dist) / 2.0
            
            # Find closest pair
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i_idx, j_idx = min_idx[0], min_idx[1]
            
            forces = np.zeros_like(pts)
            
            # Push closest pair apart
            if dists[i_idx, j_idx] > 1e-9:
                v = pts[i_idx] - pts[j_idx]
                v_norm = np.linalg.norm(v)
                v_unit = v / v_norm
                forces[i_idx] += v_unit * step_size
                forces[j_idx] -= v_unit * step_size
                
            # Boundary forces
            # Push points inward if they are close to boundary
            # We want to maximize the margin.
            # So if 1 - |c| is small, push inward.
            for k in range(N):
                r_k = r_vec[k]
                if r_k > 1e-9:
                    dist_to_bound = 1.0 - r_k
                    # If this is the limiting factor, push inward
                    # But we don't know if it's the limiting factor globally.
                    # Let's just apply a soft constraint.
                    if dist_to_bound < 2 * current_r:
                        v_in = -pts[k] / r_k
                        forces[k] += v_in * step_size * 0.5 # Weaker than pair force?
            
            pts += forces
            
            # Project to disk
            r_vec_check = np.hypot(pts[:, 0], pts[:, 1])
            mask = r_vec_check > 1.0
            if np.any(mask):
                pts[mask] /= r_vec_check[mask, np.newaxis]
                
            # If we improved, keep it? 
            # This is a greedy hill climb.
            # We should check if the move improved the radius.
            # Actually, the force direction is designed to improve.
            # But it might worsen other pairs.
            # Let's accept only if radius improves or stays same.
            # To do this properly, we need to compute r before and after.
            # We computed current_r before.
            # Let's compute new_r.
            new_r = compute_radius(pts)
            if new_r < current_r - 1e-12:
                pts -= forces # Revert
            else:
                # Keep
                pass
                
        return pts

    final_points = polish(best_points, time_limit=10.0)
    
    # Final check
    final_r = compute_radius(final_points)
    if final_r > best_r:
        best_points = final_points
        best_r = final_r

    return best_points

# EVOLVE-BLOCK-END
