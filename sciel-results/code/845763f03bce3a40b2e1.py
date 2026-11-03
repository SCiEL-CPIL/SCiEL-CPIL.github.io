import numpy as np
import time

N = 50
DEADLINE = 145.0

def compute_radius(points):
    """
    Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def initial_hexagonal():
    """Generate initial points using a hexagonal lattice pattern."""
    spacing = 0.25
    points = []
    
    max_k = 10
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    
    if len(points) >= N:
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
        
    return selected

def force_based_optimization(points, time_limit=10.0):
    """
    Optimize packing using a force-based method (Repulsion + Centering).
    This acts like a simplified Molecular Dynamics / Simulated Annealing.
    """
    points = points.copy()
    start_time = time.time()
    
    # Parameters
    k_rep = 1.0       # Repulsion strength
    k_att = 0.05      # Centering strength (pull to origin)
    damping = 0.9     # Velocity damping
    max_step = 0.05   # Max displacement per step
    
    velocities = np.zeros_like(points)
    
    while time.time() - start_time < time_limit:
        # Compute forces
        forces = np.zeros_like(points)
        
        # 1. Centering force: F = -k_att * pos
        forces -= k_att * points
        
        # 2. Repulsion force: F = k_rep * (pos_i - pos_j) / dist^3
        # Vectorized
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by 0
        
        # Force magnitude ~ 1/d^2, direction ~ diff/d -> total ~ diff/d^3
        # We want strong repulsion when close.
        # To avoid singularities, we can clip dist or use a softened potential.
        # Let's use a simple inverse square law with a floor on distance.
        safe_dists = np.maximum(dists, 0.05)
        
        # F_ij = k_rep * diff_ij / safe_dists_ij^3
        # We sum over j for each i.
        # diff is (N, N, 2). safe_dists is (N, N).
        # We need to broadcast safe_dists to (N, N, 1)
        inv_d3 = 1.0 / (safe_dists ** 3)
        inv_d3 = np.where(dists == 1.0, 0.0, inv_d3) # Zero out diagonal
        
        forces += np.sum(k_rep * diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # 3. Boundary constraint force
        # If a point is close to the boundary, push it in.
        # Simple penalty: if norm > 0.95, push towards center.
        norms = np.linalg.norm(points, axis=1)
        boundary_mask = norms > 0.95
        if np.any(boundary_mask):
            # Stronger push as it gets closer to 1.0
            # F = -k_b * (norm - 0.95) * pos/norm
            # Let's just use a strong centering force for these points
            forces[boundary_mask] -= 0.5 * points[boundary_mask] / norms[boundary_mask, np.newaxis]
        
        # Update velocities and positions
        velocities = velocities * damping + forces
        # Clip velocities to prevent explosion
        v_mags = np.linalg.norm(velocities, axis=1)
        max_v = max_step
        scale = np.where(v_mags > max_v, max_v / v_mags, 1.0)
        velocities *= scale[:, np.newaxis]
        
        points += velocities
        
        # Hard constraint: keep inside circle
        norms = np.linalg.norm(points, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            points[out_mask] = points[out_mask] / norms[out_mask, np.newaxis]
            
    return points

def annealing_loop(points, time_limit=10.0, temp_start=0.05, temp_end=0.001):
    """
    Simulated Annealing wrapper around force-based steps.
    Adds random perturbations to escape local minima.
    """
    points = points.copy()
    start_time = time.time()
    total_time = time_limit
    current_time = 0.0
    
    # We'll run force optimization for a short burst, then perturb, then repeat.
    burst_time = 0.5
    
    while current_time < total_time:
        # Run force optimization
        points = force_based_optimization(points, time_limit=burst_time)
        
        # Calculate current radius
        r_current = compute_radius(points)
        
        # Check time
        current_time = time.time() - start_time
        if current_time >= total_time:
            break
            
        # Perturbation
        # Temperature decreases over time
        progress = current_time / total_time
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Random move
        # Pick a random subset of points to move
        n_move = np.random.randint(1, 5)
        indices = np.random.choice(N, n_move, replace=False)
        
        # Gaussian perturbation
        noise = np.random.normal(0, temp, size=(n_move, 2))
        new_points = points.copy()
        new_points[indices] += noise
        
        # Ensure validity (inside circle)
        norms = np.linalg.norm(new_points, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            new_points[out_mask] = new_points[out_mask] / norms[out_mask, np.newaxis]
            
        r_new = compute_radius(new_points)
        
        # Acceptance criterion
        if r_new >= r_current:
            points = new_points
        else:
            # Accept with probability exp(-(r_current - r_new) / temp)
            # Note: r is a "score" we want to maximize, so delta = r_new - r_current is negative
            delta = r_new - r_current
            if np.random.rand() < np.exp(delta / temp):
                points = new_points
                
    return points

def construct():
    deadline = time.time() + DEADLINE
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Strategy 1: Hexagonal Init + Force Optimization
    try:
        init_pts = initial_hexagonal()
        # Initial force relaxation
        p1 = force_based_optimization(init_pts, time_limit=5.0)
        r1 = compute_radius(p1)
        if r1 > best_overall_r:
            best_overall_r = r1
            best_overall_points = p1
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()
        
    # Strategy 2: Annealing on top of best so far
    try:
        if best_overall_points is None:
            base_pts = initial_hexagonal()
        else:
            base_pts = best_overall_points
            
        # Run annealing for the remaining time (minus safety margin)
        remaining_time = max(0, (deadline - 5) - time.time())
        if remaining_time > 1.0:
            p2 = annealing_loop(base_pts, time_limit=remaining_time)
            r2 = compute_radius(p2)
            if r2 > best_overall_r:
                best_overall_r = r2
                best_overall_points = p2
    except Exception:
        pass

    # Final check
    if best_overall_points is not None:
        # One last quick force relaxation to clean up
        if time.time() < deadline - 1:
            p_final = force_based_optimization(best_overall_points, time_limit=1.0)
            r_final = compute_radius(p_final)
            if r_final > best_overall_r:
                best_overall_r = r_final
                best_overall_points = p_final
                
    return best_overall_points if best_overall_points is not None else initial_hexagonal()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
