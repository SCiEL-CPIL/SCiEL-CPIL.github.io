import numpy as np
import time

N = 200

def _hex_init():
    """Generate a hexagonal lattice initialization for N circles."""
    # Hexagonal lattice points
    # We want to fit N points in [-0.5, 0.5]^2
    # Approximate side length of hexagon cell: 2/r where r is radius.
    # Let's just generate a large grid and pick the N closest to center?
    # Or generate a grid that fits.
    
    # A standard hex grid:
    # x = i * dx + (j % 2) * dx/2
    # y = j * dy
    # dx = 1.0, dy = sqrt(3)/2 * dx
    # Let's aim for a spacing that results in ~200 points.
    # Area of square = 1. Area per point ~ 1/200 = 0.005.
    # Hex cell area = (sqrt(3)/2) * s^2. s^2 ~ 0.005 * 2/sqrt(3) ~ 0.0058. s ~ 0.076.
    # Spacing between centers ~ s.
    
    spacing = 0.075
    dx = spacing
    dy = spacing * np.sqrt(3) / 2.0
    
    points = []
    # Generate points in a range slightly larger than [-0.5, 0.5]
    # i, j range
    max_i = int(1.0 / dx) + 2
    max_j = int(1.0 / dy) + 2
    
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = i * dx + (j % 2) * dx / 2.0
            y = j * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, pick the N closest to the center to maintain symmetry?
    # Or just take the first N? Taking closest to center is better for packing.
    if len(points) > N:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    elif len(points) < N:
        # If we have too few, add random points or shrink spacing and retry
        # For robustness, just add random points inside
        extra = np.random.uniform(-0.5, 0.5, (N - len(points), 2))
        points = np.vstack([points, extra])
        
    return points

def _soft_min_value(centers, p):
    """Calculate the soft-min of distances."""
    # Distances to boundary
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # This is min(0.5 - |x|, 0.5 - |y|)
    dists_b = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists_p = np.sqrt(dists_sq)
    
    # We want min of all these distances.
    # Soft min: (sum(d^-p))^-1/p
    # To avoid division by zero, clip distances
    eps = 1e-6
    dists_b = np.maximum(dists_b, eps)
    dists_p = np.maximum(dists_p, eps)
    
    # Sum of inverse powers
    sum_b = np.sum(dists_b ** (-p))
    sum_p = np.sum(dists_p ** (-p))
    
    total_sum = sum_b + sum_p
    val = total_sum ** (-1.0 / p)
    
    return val

def _soft_min_gradient(centers, p):
    """Calculate the gradient of the soft-min value with respect to centers."""
    grad = np.zeros_like(centers)
    eps = 1e-6
    
    # 1. Boundary terms
    # f_b = (sum_i (d_{b,i})^-p)^-1/p
    # d/dx f_b = -1/p * f_b^(1+p) * sum_j (d_{b,j})^(-p-1) * d(d_{b,j})/dx
    # This is complex to vectorize fully. Let's do it per point for boundary.
    
    dists_b = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    dists_b = np.maximum(dists_b, eps)
    
    sum_b = np.sum(dists_b ** (-p))
    total_sum = sum_b + np.sum(np.maximum(np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2) - np.eye(N)*np.inf), eps) ** (-p))
    
    # Actually, computing the full gradient analytically is prone to error.
    # Given the time, a simpler **Repulsion Force** model is often sufficient and faster to implement correctly.
    # Force on i from j: if dist < D, push i away from j.
    # Force from boundary: if dist < D_b, push i inwards.
    # This is essentially gradient ascent on a potential energy that is high when constraints are violated.
    
    return grad # Placeholder, will replace with force model in main loop

def construct():
    """
    Packs N equal circles in a unit square [-0.5, 0.5]^2 to maximize radius.
    Uses Hexagonal Init + Force-Directed Relaxation.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # 1. Initialization
    centers = _hex_init()
    best_centers = centers.copy()
    best_radius = 0.0
    
    def calc_radius(centers):
        """Calculate the maximum possible radius for the given centers."""
        dists_b = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
        min_boundary = np.min(dists_b)
        if min_boundary < 0:
            return 0.0
            
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dists_sq))
        
        return min(min_boundary, min_pair_dist / 2.0)
    
    current_radius = calc_radius(centers)
    if current_radius > best_radius:
        best_radius = current_radius
        best_centers = centers.copy()
        
    # 2. Optimization: Force-Directed Relaxation
    # We simulate particles repelling each other and the walls.
    # We slowly increase the "hardness" of the repulsion or the target distance.
    
    # Parameters
    damping = 0.5
    steps_per_iter = 500
    
    # Target radius progression
    # Start with a small radius, increase it.
    # The best known is ~0.0366.
    # Let's try to push towards this.
    
    # Instead of fixed target, we can use a "soft" potential:
    # V = sum_{i<j} max(0, R_target - d_ij)^2 + sum_i max(0, R_target - d_bi)^2
    # Gradient descent on V.
    # We increase R_target over time.
    
    r_target = best_radius * 0.9
    max_r = 0.037 # Slightly above best known
    
    current_r = best_radius
    
    while time.time() - start_time < time_limit:
        # Increase target radius slightly
        r_target = min(r_target * 1.01, max_r)
        
        # Run relaxation steps for this target
        for step in range(steps_per_iter):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            # We only need to consider pairs within 2*r_target + margin
            # For N=200, O(N^2) is fine.
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Mask for pairs too close
            mask = dists < 2.0 * r_target
            
            # Calculate forces
            # F_ij = k * (2*r_target - d_ij) * unit_vec_ij
            # Only if d_ij < 2*r_target
            k = 1.0 # Spring constant
            # Vectorized calculation
            # dists is N x N
            # We want to sum forces on i from all j
            # F_i = sum_j k * (2*r_target - d_ij) * (c_i - c_j)/d_ij
            # Note: (c_i - c_j)/d_ij is the unit vector from j to i.
            
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-6, dists, 1e-6)
            
            # Magnitude of force from j on i
            # mag_ij = k * (2*r_target - d_ij) if d_ij < 2*r_target else 0
            mag = np.where(mask, k * (2.0 * r_target - dists), 0.0)
            
            # Direction is (c_i - c_j) / d_ij
            # We can compute this as:
            # dir_ij = diff / dists_safe[:,:,np.newaxis]
            # Force_i = sum_j mag_ij * dir_ij
            
            # To save memory and time, we can just do the loop if vectorized is too slow, 
            # but N=200 is small.
            dir_vec = diff / dists_safe[:, :, np.newaxis]
            # Zero out directions where mask is False
            dir_vec *= mask[:, :, np.newaxis]
            
            # Sum over j (axis 1)
            forces += np.sum(mag[:, :, np.newaxis] * dir_vec, axis=1)
            
            # Boundary forces
            # Push inwards if too close to boundary
            # dist_b = min(0.5 - |x|, 0.5 - |y|)
            # If dist_b < r_target, push towards center
            # Force direction: towards center (0,0) ? No, towards the valid region.
            # If x > 0.5 - r_target, force is negative x.
            # If x < -(0.5 - r_target), force is positive x.
            
            margin = r_target
            # X forces
            # If x > 0.5 - margin, Fx = -k_b * (x - (0.5 - margin))
            # If x < -(0.5 - margin), Fx = k_b * (x - (-(0.5 - margin))) = k_b * (x + 0.5 - margin)
            
            k_b = 1.0
            x = centers[:, 0]
            y = centers[:, 1]
            
            # Right wall
            mask_rx = x > 0.5 - margin
            forces[mask_rx, 0] -= k_b * (x[mask_rx] - (0.5 - margin))
            
            # Left wall
            mask_lx = x < -(0.5 - margin)
            forces[mask_lx, 0] += k_b * (x[mask_lx] - (-(0.5 - margin)))
            
            # Top wall
            mask_ty = y > 0.5 - margin
            forces[mask_ty, 1] -= k_b * (y[mask_ty] - (0.5 - margin))
            
            # Bottom wall
            mask_by = y < -(0.5 - margin)
            forces[mask_by, 1] += k_b * (y[mask_by] - (-(0.5 - margin)))
            
            # Update positions
            # Simple Euler integration with damping
            centers += damping * forces
            centers = np.clip(centers, -0.5, 0.5)
            
            # Check radius every few steps
            if step % 10 == 0:
                current_radius = calc_radius(centers)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
                    
        # If we didn't improve much, maybe we are stuck in a local minimum?
        # The increasing r_target should help escape.
        # If r_target is maxed out and we are not improving, we can try a small random shake.
        if r_target >= max_r:
            # Shake
            if np.random.random() < 0.1:
                idxs = np.random.choice(N, 10, replace=False)
                centers[idxs] += np.random.normal(0, 0.01, (10, 2))
                centers = np.clip(centers, -0.5, 0.5)
                current_radius = calc_radius(centers)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
            
            # If stuck at max_r for a while, break?
            # For now, just keep going until time limit.
            
    return best_centers
