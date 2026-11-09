"""Physics-based relaxation for 300 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    """
    Place 300 equal circles inside the unit circle to maximize the common radius.
    Uses physics-based relaxation: initialize on hexagonal lattice, then optimize
    using gradient ascent on a penalty-based energy that enforces non-overlap
    and boundary containment.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|)).
    We maximize this r by iteratively pushing apart overlapping circles and
    pulling inward circles that are too close to the boundary.
    """
    np.random.seed(42)
    
    # Step 1: Initialize on a hexagonal lattice
    # Estimate spacing for 300 circles in unit circle
    # Area of unit circle = pi, so average area per circle ~ pi/300
    # For hexagonal packing, area per circle ~ (sqrt(3)/2) * d^2 where d is spacing
    # So d ~ sqrt(2*pi/(300*sqrt(3)/2)) = sqrt(4*pi/(300*sqrt(3)))
    d_est = np.sqrt(4 * np.pi / (N * np.sqrt(3)))
    
    # Generate hexagonal lattice points
    points = []
    k_max = int(1.0 / d_est) + 3
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * d_est
            y = j * d_est * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - d_est / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback: use a simpler grid
        points = np.random.randn(N, 2) * 0.5
    else:
        # Select N points closest to origin for better initial packing
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:N]
        points = points[idx]
    
    # Step 2: Define the objective function to maximize
    # We want to maximize r such that all circles are non-overlapping and inside
    # Score: r = min(min_pair_dist/2, min_boundary_dist)
    # We'll use a soft penalty approach with iterative optimization
    
    def compute_score(pts):
        """Compute the current score (radius) for a configuration."""
        # Boundary constraint
        dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dist_to_boundary)
        
        # Pairwise distances
        # Use efficient computation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)
    
    def energy(pts, r_target):
        """
        Energy function: penalize violations of constraints.
        We want all pairwise distances >= 2*r and all points within distance (1-r) from origin.
        """
        # Boundary violations
        dist_to_origin = np.linalg.norm(pts, axis=1)
        boundary_violations = np.maximum(0, dist_to_origin - (1.0 - r_target))
        boundary_energy = np.sum(boundary_violations**2)
        
        # Pairwise overlap violations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Penalty for pairs closer than 2*r_target
        overlap = np.maximum(0, 2 * r_target - dists)
        pair_energy = np.sum(overlap**2)
        
        return boundary_energy + pair_energy
    
    def energy_grad(pts, r_target):
        """Gradient of the energy function."""
        grad = np.zeros_like(pts)
        
        # Boundary gradient
        dist_to_origin = np.linalg.norm(pts, axis=1)
        mask = dist_to_origin > (1.0 - r_target)
        if np.any(mask):
            # d/dx max(0, |x| - (1-r))^2 = 2*(|x|-(1-r)) * x/|x| for |x| > (1-r)
            viol = dist_to_origin[mask] - (1.0 - r_target)
            norm_mask = dist_to_origin[mask]
            grad[mask] += 2 * viol[:, np.newaxis] * pts[mask] / norm_mask[:, np.newaxis]
        
        # Pairwise gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # For pairs where dist < 2*r, gradient contribution
        mask = dists < 2 * r_target
        # grad_i += sum_j 2*(2r - d_ij) * (p_i - p_j)/d_ij
        for i in range(N):
            for j in range(N):
                if i != j and mask[i, j]:
                    d = dists[i, j]
                    if d > 1e-10:
                        grad[i] += 2 * (2 * r_target - d) * diff[i, j] / d
        
        return grad
    
    # Step 3: Iterative optimization
    # Start with a small radius and increase it while relaxing the configuration
    current_score = compute_score(points)
    r = current_score * 0.9  # Start slightly below current score
    
    # Use a simple gradient ascent with adaptive step size
    best_points = points.copy()
    best_score = current_score
    
    # Pre-compute some constants
    max_iter = 5000
    step_size = 0.01
    
    for iteration in range(max_iter):
        # Compute gradient
        grad = energy_grad(points, r)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Take a step
        points_new = points + step_size * grad
        
        # Ensure points stay within a reasonable region
        norms = np.linalg.norm(points_new, axis=1)
        mask_out = norms > 1.5
        if np.any(mask_out):
            points_new[mask_out] = points_new[mask_out] / norms[mask_out, np.newaxis] * 1.4
        
        # Compute new score
        new_score = compute_score(points_new)
        
        if new_score > best_score:
            best_score = new_score
            best_points = points_new.copy()
        
        # Try to increase r if we're making progress
        if iteration % 100 == 0 and new_score > r * 0.95:
            r = min(new_score * 0.95, 0.06)
            step_size = max(0.001, step_size * 0.99)
        else:
            step_size = max(0.001, step_size * 0.999)
        
        points = points_new
    
    # Step 4: Final refinement with smaller steps
    points = best_points.copy()
    r = compute_score(points) * 0.99
    step_size = 0.001
    
    for iteration in range(2000):
        grad = energy_grad(points, r)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        points_new = points + step_size * grad
        
        norms = np.linalg.norm(points_new, axis=1)
        mask_out = norms > 1.5
        if np.any(mask_out):
            points_new[mask_out] = points_new[mask_out] / norms[mask_out, np.newaxis] * 1.4
        
        new_score = compute_score(points_new)
        
        if new_score > best_score:
            best_score = new_score
            best_points = points_new.copy()
        
        if iteration % 200 == 0:
            r = min(compute_score(points_new) * 0.98, 0.06)
            step_size *= 0.999
        
        points = points_new
    
    return best_points

# EVOLVE-BLOCK-END
