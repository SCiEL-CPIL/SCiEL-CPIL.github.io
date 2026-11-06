"""Pack 300 equal circles in a unit disk using Fibonacci init and incremental radius optimization."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fibonacci_disk(n, radius=1.0):
    """
    Generate n points uniformly distributed in a disk of given radius
    using the Fibonacci spiral method.
    """
    # Golden angle
    angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Indices 0 to n-1
    k = np.arange(n)
    
    # Radius distribution: sqrt(k/n) ensures uniform area density
    # We scale to fit inside radius=1.0
    # Note: k=0 is at center.
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def _compute_min_radius(points):
    """
    Compute the maximum possible radius for the given points.
    """
    if len(points) == 0:
        return 0.0
    
    # Distance to boundary
    dists_to_center = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - dists_to_center)
    
    if boundary_min < 0:
        return 0.0 # Invalid, points outside
    
    # Pairwise distances
    # For N=300, 300x300 matrix is fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    pair_min = min_pair_dist / 2.0
    
    return min(boundary_min, pair_min)

def _relax_for_radius(points, target_r, max_iter=100, lr=0.01):
    """
    Attempt to adjust points to fit circles of radius target_r.
    Minimizes the penalty function for overlaps and boundary violations.
    Returns success (bool) and new points.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    
    # Penalty function: sum of squared violations
    # Violation for pair i,j: max(0, 2*target_r - dist_ij)^2
    # Violation for point i: max(0, |c_i| + target_r - 1)^2
    
    # We use a simple gradient descent on this penalty.
    # Gradient of pair penalty w.r.t c_i:
    # if d < 2r: 2 * (2r - d) * (-(c_i - c_j) / d) = -2(2r-d)/d * (c_i - c_j)
    # Wait, derivative of (2r - d)^2 is 2(2r-d)(-1) * d(d)/dc_i.
    # d(d)/dc_i = (c_i - c_j) / d.
    # So grad = -2(2r-d) * (c_i - c_j) / d.
    # This is a repulsive force pushing i away from j if d < 2r.
    
    # Gradient of boundary penalty w.r.t c_i:
    # if |c_i| > 1 - target_r: 2 * (|c_i| + target_r - 1) * (c_i / |c_i|)
    # This pushes i inward.
    
    current_penalty = 1e10
    for it in range(max_iter):
        forces = np.zeros_like(points)
        
        # Pairwise forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Identify pairs that are overlapping or close
        # To avoid division by zero and heavy computation, only consider d < 2*target_r * 1.1
        mask = dist_matrix < 2.2 * target_r
        np.fill_diagonal(mask, False)
        
        # Compute forces for active pairs
        # force_i += sum_j -2 * (2r - d_ij) * (c_i - c_j) / d_ij
        # We can vectorize this using the mask.
        
        # Create a weighted direction matrix
        # W_ij = -2 * (2r - d_ij) / d_ij if d_ij < 2r else 0
        # Note: if d_ij >= 2r, force is 0.
        
        # To be efficient, we only compute for masked elements.
        # But numpy broadcasting for 300x300 is fast.
        
        # Let's compute the term (2r - d) / d
        # To avoid div by zero, use where
        inv_d = np.zeros_like(dist_matrix)
        np.fill_diagonal(inv_d, 0)
        valid_d = dist_matrix > 1e-10
        inv_d[valid_d] = 1.0 / dist_matrix[valid_d]
        
        # Overlap factor
        overlap = np.maximum(0, 2.0 * target_r - dist_matrix)
        
        # Force magnitude coefficient
        # grad = -2 * overlap * inv_d * (c_i - c_j)
        # Note: this is for the penalty (2r-d)^2.
        # The force on i is sum_j -2 * (2r - d_ij) * (c_i - c_j) / d_ij
        
        coeff = -2.0 * overlap * inv_d
        np.fill_diagonal(coeff, 0)
        
        # forces = coeff @ (c_i - c_j) ... wait.
        # coeff is a matrix M.
        # Force on i = sum_j M_ij * (c_i - c_j)
        # Force on i = sum_j M_ij * c_i - sum_j M_ij * c_j
        # Force on i = (sum_j M_ij) * c_i - (M @ c)_i
        
        row_sums = np.sum(coeff, axis=1, keepdims=True)
        forces_pair = row_sums * points - (coeff @ points)
        
        # Boundary forces
        dists = np.linalg.norm(points, axis=1)
        boundary_violation = np.maximum(0, dists + target_r - 1.0)
        
        # grad = 2 * violation * (c_i / |c_i|)
        # If dist is 0, direction is undefined, but violation is target_r - 1 < 0 usually?
        # If target_r < 1, and dist=0, violation is max(0, target_r - 1) = 0.
        # So center point has no boundary force unless target_r > 1.
        
        dir_vec = np.zeros_like(points)
        valid_dist = dists > 1e-10
        dir_vec[valid_dist] = points[valid_dist] / dists[valid_dist, np.newaxis]
        
        forces_boundary = 2.0 * boundary_violation[:, np.newaxis] * dir_vec
        
        total_forces = forces_pair + forces_boundary
        
        # Update
        points += lr * total_forces
        
        # Hard constraint: keep inside disk
        new_dists = np.linalg.norm(points, axis=1)
        outside_mask = new_dists > 1.0
        if np.any(outside_mask):
            points[outside_mask] *= (1.0 / new_dists[outside_mask, np.newaxis]) * 0.999
            
        # Check penalty
        # Calculate min distance to see if we are making progress
        # We don't need full penalty every step, just check if we converged
        # For simplicity, we run max_iter steps.
        
    # Check if the configuration is valid for target_r
    final_r = _compute_min_radius(points)
    if final_r >= target_r - 1e-9:
        return True, points
    else:
        return False, points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Initialize with Fibonacci disk
    points = _fibonacci_disk(N)
    current_r = _compute_min_radius(points)
    
    # 2. Initial relaxation to settle the configuration
    # Try to fit a slightly larger radius than the initial one
    # The initial Fibonacci distribution is uniform, so min dist is roughly ~ 1/sqrt(N)
    # Let's do a few iterations of relaxation with a small target increase
    
    # Set a base learning rate
    base_lr = 0.01
    
    # Loop: try to increase the radius
    # We use a "hill climbing" approach on the radius.
    # Step: try r_new = r * (1 + delta)
    # If relaxation succeeds, accept. Else, reduce delta or stop.
    
    delta = 0.001
    min_delta = 1e-6
    stall_count = 0
    
    best_points = points.copy()
    best_r = current_r
    
    while time.time() < deadline - 5.0:
        target_r = best_r * (1.0 + delta)
        
        # Run relaxation
        success, new_points = _relax_for_radius(best_points, target_r, max_iter=50, lr=base_lr)
        
        if success:
            best_points = new_points
            best_r = _compute_min_radius(best_points)
            stall_count = 0
            # Optionally increase delta slightly if we are fast
            # delta = min(delta * 1.1, 0.01)
        else:
            # Failed to find a configuration for this target_r
            # Reduce step size
            delta *= 0.5
            if delta < min_delta:
                # Try to polish the current best_r with smaller steps?
                # Or break if we can't improve
                # Let's try one more pass with a very small delta to see if it's numerical noise
                if delta < 1e-8:
                    break
                stall_count += 1
                if stall_count > 10:
                    break
    
    # Final polish: run a few more iterations with the current best_r to ensure stability
    # This helps if the last successful step left points slightly sub-optimal
    _relax_for_radius(best_points, best_r, max_iter=20, lr=base_lr * 0.5)
    
    # Verify final score
    final_score = _compute_min_radius(best_points)
    if final_score > best_r:
        best_r = final_score
        
    return best_points
# EVOLVE-BLOCK-END
