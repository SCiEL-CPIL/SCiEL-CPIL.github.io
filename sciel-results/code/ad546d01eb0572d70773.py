"""Improved circle packing for N=300 in unit circle using multi-restart gradient optimization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_spiral_init():
        """Generate a golden-angle spiral configuration."""
        # Golden angle
        ga = np.pi * (3.0 - np.sqrt(5.0))
        indices = np.arange(N)
        # Radius scaling: max radius should be < 1
        # r_i = sqrt(i/N) * R_max
        # To allow space for circle radius r, R_max should be ~ 1 - 2*r_est
        # r_est ~ 0.05, so R_max ~ 0.9
        R_max = 0.9
        radii = np.sqrt(indices / N) * R_max
        angles = indices * ga
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.column_stack((x, y))
        return pts

    def optimize_barrier(pts, max_iters=5000, deadline_check=None):
        """
        Optimize using a log-barrier objective function.
        Objective: Maximize sum(log(d_ij)) + sum(log(1-|c_i|))
        This is equivalent to maximizing the product of distances and boundary margins.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Adaptive step size
        step_size = 0.01
        # Learning rate for step size adjustment
        lr_decay = 0.99
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute pairwise differences and distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            # Compute gradients of the log-barrier objective
            # 1. Pairwise term: sum_{j != i} d/dc_i log(d_ij) = sum_{j != i} (c_i - c_j) / d_ij^2
            # Note: d/dc_i log(d_ij) = (c_i - c_j) / d_ij^2
            # We have diff = c_i - c_j. So grad_pair = sum_j diff_ij / dists_ij^2
            # dists_sq is already computed.
            # Avoid division by zero (diagonal is inf, so 1/inf = 0)
            inv_dists_sq = 1.0 / dists_sq
            # Mask diagonal explicitly just in case
            np.fill_diagonal(inv_dists_sq, 0.0)
            
            # grad_pair[i] = sum_j (c_i - c_j) / d_ij^2
            # This is a matrix multiplication: diff @ inv_dists_sq
            # diff is (N, N, 2), inv_dists_sq is (N, N)
            # We want sum over j for each i.
            # diff[i, j, :] * inv_dists_sq[i, j]
            # Sum over axis 1.
            grad_pair = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
            
            # 2. Boundary term: sum_i d/dc_i log(1 - |c_i|)
            # Let r_i = |c_i|. d/dc_i log(1 - r_i) = -1/(1 - r_i) * d/dc_i r_i
            # d/dc_i r_i = c_i / r_i
            # So grad_bound[i] = -c_i / (r_i * (1 - r_i))
            radii = np.sqrt(dists_sq.diagonal()) # Wait, dists_sq diagonal is inf. 
            # Recompute radii properly
            radii = np.hypot(pts[:, 0], pts[:, 1])
            # Avoid division by zero for center point
            radii_safe = np.where(radii > 1e-10, radii, 1e-10)
            # Term: 1 / (r * (1 - r))
            # If r is close to 1, this becomes very large (barrier).
            # If r is 0, we use safe radius.
            bound_factor = 1.0 / (radii_safe * (1.0 - radii_safe))
            grad_bound = -pts * bound_factor[:, np.newaxis]
            
            # Total gradient
            grad = grad_pair + grad_bound
            
            # Update points
            pts += grad * step_size
            
            # Project inside unit circle (hard constraint to maintain validity)
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
            
            # Evaluate score
            current_score = compute_score(pts)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                # Increase step size slightly if improving
                step_size = min(step_size * 1.05, 0.05)
            else:
                # Decrease step size if not improving
                step_size *= lr_decay
                
                # If step size is very small, try a random kick to escape local minima
                if step_size < 1e-5:
                    if np.random.random() < 0.1:
                        pts += np.random.randn(N, 2) * 0.005
                        new_radii = np.hypot(pts[:, 0], pts[:, 1])
                        outside = new_radii > 1.0
                        if np.any(outside):
                            pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
                    step_size = 0.01 # Reset step size
            
        return best_pts, best_score

    # Main execution
    # 1. Initialize with golden-angle spiral
    pts = generate_spiral_init()
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # 2. Barrier optimization
    pts, score = optimize_barrier(pts, max_iters=10000, deadline_check=deadline - 5)
    if score > best_score:
        best_score = score
        best_pts = pts
    
    # 3. Final refinement with local search (optional, if time permits)
    # The barrier method should be quite effective. 
    # We can add a simple local search if time remains.
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Simple local search: perturb points with low boundary margin or low min distance
        for _ in range(int(remaining_time * 100)): # Approximate iterations
            if time.time() > deadline - 2:
                break
            # Find points with lowest "freedom"
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            boundary_margin = 1.0 - radii
            
            # We don't want to recompute all pairwise distances every iteration here if it's too slow
            # But N=300 is small.
            # Let's just do a few random perturbations
            idx = np.random.randint(N)
            orig = best_pts[idx].copy()
            best_pts[idx] += np.random.randn(2) * 0.005
            r = np.hypot(best_pts[idx, 0], best_pts[idx, 1])
            if r > 1.0:
                best_pts[idx] = best_pts[idx] / r
            
            new_score = compute_score(best_pts)
            if new_score > best_score:
                best_score = new_score
            else:
                best_pts[idx] = orig
    
    # Final validation
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
