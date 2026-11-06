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

    best_r = compute_radius(best_points)
    
    # Optimization Loop
    # Strategy: Target-aware force relaxation.
    # We only apply forces that correct violations of the current best_r (or a slightly larger target).
    # This allows the points to "settle" into a configuration that is feasible for r_target.
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 50
    
    # Force scaling parameters
    lambda_pair = 1.0
    lambda_bound = 1.0
    
    for it in range(200000):
        if time.time() > deadline:
            break
            
        # Compute current radius periodically to update best_r and target
        # Doing it every iteration is feasible for N=300 with vectorized ops
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation to escape local minimum
                # Targeted perturbation: move points involved in the minimum distance or boundary violation
                radii = np.hypot(points[:, 0], points[:, 1])
                
                # Find points closest to boundary
                boundary_proximity = 1.0 - radii
                min_bound_val = np.min(boundary_proximity)
                
                # Find closest pairs
                # Reuse diff/dists from compute_radius? No, we didn't store it.
                # For perturbation, a simple random kick to the "most constrained" points is effective.
                
                # Identify points with small boundary clearance
                close_to_boundary = boundary_proximity < (min_bound_val + 0.005 * best_r)
                
                # Identify points in close pairs (approximate by just picking random few if we don't want to recompute full matrix)
                # Let's just perturb a random subset of points that are "tight"
                num_perturb = 10
                if np.sum(close_to_boundary) > 0:
                    # Perturb some boundary points
                    idxs = np.where(close_to_boundary)[0]
                    chosen = np.random.choice(idxs, size=min(num_perturb, len(idxs)), replace=False)
                    points[chosen] += np.random.normal(0, 0.005 * best_r, (len(chosen), 2))
                
                # Also perturb some random points to help global rearrangement
                random_idx = np.random.choice(N, size=num_perturb, replace=False)
                points[random_idx] += np.random.normal(0, 0.003 * best_r, (num_perturb, 2))
                
                # Ensure validity after perturbation
                new_radii = np.hypot(points[:, 0], points[:, 1])
                outside = new_radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
                    
                no_improve_count = 0
                # Reduce step size after perturbation to allow settling
                step_size *= 0.8

        # Compute Forces
        # Target radius: slightly larger than best_r to encourage improvement
        # If we use best_r, we might stall. Using best_r * 1.001 pushes for improvement.
        # However, if we push too hard, we might create large violations.
        # Let's use r_target = best_r * 1.005
        r_target = best_r * 1.005
        
        forces = np.zeros_like(points)
        
        # 1. Pairwise Forces
        # Only apply if d_ij < 2 * r_target
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Mask for active pairs: dists < 2 * r_target
        active_mask = dists < (2.0 * r_target)
        
        # Force magnitude: proportional to violation / d^2
        # Violation = 2*r_target - d
        # We want to push apart.
        # F_ij = k * (2*r_target - d) / d^2 * direction
        # direction = (p_i - p_j) / d
        # So F_ij = k * (2*r_target - d) / d^3 * (p_i - p_j)
        
        # To avoid division by zero or small numbers, we can use a safe denominator.
        # Since active_mask ensures d < 2*r_target, d is not zero (unless points coincide, which is bad).
        # If points coincide, d=0. We should handle this.
        
        # Calculate factors for active pairs
        # factor_ij = (2*r_target - d) / d^3
        # We only compute this for active pairs to save time? 
        # For N=300, full matrix is fine.
        
        # Initialize factors to 0
        factors = np.zeros((N, N))
        
        # Only compute where active
        if np.any(active_mask):
            # d_safe to avoid div by 0
            d_safe = np.where(dists > 1e-6, dists, 1e-6)
            # violation = 2*r_target - d
            # If d is very small, violation is ~2*r_target.
            # factor = violation / d^3
            violation = 2.0 * r_target - dists
            # Clip violation to be non-negative (should be positive by mask, but just in case)
            violation = np.maximum(violation, 0)
            factors[active_mask] = violation[active_mask] / (d_safe[active_mask] ** 3)
            
            # Apply symmetry: factors[i,j] = factors[j,i]
            # The formula is symmetric.
            
            # Force on i from j: factor[i,j] * (p_i - p_j)
            # F_i = sum_j factor[i,j] * diff[i,j]
            pair_forces = np.einsum('ij,ijk->ik', factors, diff)
            forces += lambda_pair * pair_forces

        # 2. Boundary Forces
        # Only apply if 1 - |c_i| < r_target
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_violation = r_target - (1.0 - radii)
        active_boundary = boundary_violation > 0
        
        if np.any(active_boundary):
            # Force direction: inward, i.e., -p_i / |p_i|
            # Magnitude: proportional to violation
            # F_i = k * violation * (-p_i / |p_i|)
            # Let's use a stronger penalty if close to boundary
            # F_i = k * (violation)^2 * (-p_i / |p_i|) ? 
            # Linear is smoother. Let's stick to linear scaled by 1/r_target to normalize.
            
            # Safe radii for division
            r_safe = np.where(radii > 1e-6, radii, 1e-6)
            # Normal vector inward
            norm_vec = -points / r_safe[:, np.newaxis]
            
            # Scale factor
            # We want the force to be comparable to pair forces.
            # Pair force magnitude ~ (2r)/d^3 * d ~ 2r/d^2. At d~2r, ~0.5.
            # Boundary force ~ violation. If violation is small, force is small.
            # Let's scale by 1/r_target to make it dimensionless-ish.
            bound_factors = boundary_violation / r_target
            bound_factors[~active_boundary] = 0.0
            
            bound_forces = bound_factors[:, np.newaxis] * norm_vec
            forces += lambda_bound * bound_forces

        # Update positions
        points += forces * step_size
        
        # Enforce hard constraint: inside unit circle
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            # Project back inside
            points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
            
        # Decay step size
        step_size *= 0.9999

        # Optional: If we reach a very high radius, we can break early to save time
        # Best known is ~0.05315. 
        if best_r > 0.0532:
             # Do a few more iterations to polish
             pass

    # Final Polish Phase
    # Use a smaller step size and focus on the current best configuration
    if time.time() < deadline - 5.0:
        points = best_points.copy()
        polish_deadline = deadline - 2.0
        polish_step = step_size * 0.5
        
        for it in range(50000):
            if time.time() > polish_deadline:
                break
            
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
            
            r_target = best_r * 1.001
            
            forces = np.zeros_like(points)
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            active_mask = dists < (2.0 * r_target)
            factors = np.zeros((N, N))
            if np.any(active_mask):
                d_safe = np.where(dists > 1e-6, dists, 1e-6)
                violation = 2.0 * r_target - dists
                violation = np.maximum(violation, 0)
                factors[active_mask] = violation[active_mask] / (d_safe[active_mask] ** 3)
                pair_forces = np.einsum('ij,ijk->ik', factors, diff)
                forces += lambda_pair * pair_forces

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
                
            polish_step *= 0.99995

    return best_points
