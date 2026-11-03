import numpy as np
import time

N = 200

def construct():
    # 1. Initialization: Hexagonal grid with jitter
    # We use the best hex grid from the previous logic as a starting point
    # but we will treat it as a seed for the physics simulation.
    
    # Find best hex grid parameters
    best_init = None
    best_init_score = -1
    
    for m in range(10, 22):
        base = N // m
        rem = N % m
        row_counts = [base + 1] * rem + [base] * (m - rem)
        n_max = max(row_counts)
        dx_width = 1.0 / n_max
        dx_height = 1.0 / (1.0 + (m - 1) * np.sqrt(3) / 2.0)
        dx = min(dx_width, dx_height)
        dy = dx * np.sqrt(3) / 2.0
        r_est = dx / 2.0
        if r_est > best_init_score:
            best_init_score = r_est
            best_init = (m, row_counts, dx, dy)
            
    m, row_counts, dx, dy = best_init
    
    centers = []
    y_start = -((m - 1) * dy) / 2.0
    for j, n in enumerate(row_counts):
        y = y_start + j * dy
        x_start = -((n - 1) * dx) / 2.0
        for i in range(n):
            x = x_start + i * dx
            centers.append([x, y])
            
    P = np.array(centers, dtype=float)
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    P += np.random.randn(N, 2) * 0.001
    
    # Ensure initial P is valid (inside square)
    P[:, 0] = np.clip(P[:, 0], -0.49, 0.49)
    P[:, 1] = np.clip(P[:, 1], -0.49, 0.49)
    
    def get_min_radius(P):
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        bound_dists = np.min([
            0.5 - P[:, 0],
            0.5 + P[:, 0],
            0.5 - P[:, 1],
            0.5 + P[:, 1]
        ], axis=0)
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)
    
    current_r = get_min_radius(P)
    
    start_time = time.time()
    max_time = 110.0
    
    # Parameters for physics simulation
    alpha = 0.005 # Step size
    noise_scale = 0.001
    
    # Iterative Radius Expansion
    # We try to increase the target radius and relax the system to satisfy it.
    target_r = current_r
    success_count = 0
    fail_count = 0
    epsilon = 0.0005 # Relative increase factor
    
    while time.time() - start_time < max_time:
        # Propose a new target radius
        proposed_r = target_r * (1.0 + epsilon)
        
        # Run relaxation to satisfy constraints for proposed_r
        # Constraints: d_ij >= 2 * proposed_r, b_i >= proposed_r
        min_dist_req = 2.0 * proposed_r
        
        # Relaxation Loop
        # We run a fixed number of steps or until convergence/stability
        for _ in range(200):
            forces = np.zeros((N, 2))
            
            # Pairwise forces
            # O(N^2) calculation
            diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=-1))
            np.fill_diagonal(dists, 1.0) # Avoid div by 0, will be masked
            
            # Only apply force if dist < min_dist_req
            # Force magnitude: (min_dist_req - dist)
            # Direction: diffs / dist
            # F_ij on i is proportional to (min_dist_req - dist) * (P_i - P_j)/dist
            # Note: diffs[i,j] = P_i - P_j
            
            # Create mask for active pairs
            mask = dists < min_dist_req
            np.fill_diagonal(mask, False)
            
            # Calculate forces
            # We need to sum forces from all j to i
            # F_i = sum_j (min_dist_req - d_ij) * (P_i - P_j) / d_ij
            # Let's compute vectorized
            # term = (min_dist_req - dists) * mask
            # term = term / dists
            # F = sum_j term[:, j] * diffs[:, j, :]
            
            # To save memory/time, we can do this in chunks or just rely on numpy efficiency
            # N=200 is small enough for full matrix
            
            weights = np.zeros_like(dists)
            weights[mask] = (min_dist_req - dists[mask]) / dists[mask]
            
            # Force on i is sum over j of weights[i,j] * (P[i] - P[j])
            # This is equivalent to:
            # F_i = sum_j weights[i,j] * P[i] - sum_j weights[i,j] * P[j]
            # F_i = P[i] * sum_j weights[i,j] - sum_j (weights[i,j] * P[j])
            
            row_sums = np.sum(weights, axis=1)
            # weighted_sum_P = weights @ P
            weighted_P = np.dot(weights, P)
            
            forces += P * row_sums[:, np.newaxis] - weighted_P
            
            # Wall forces
            # Left wall: x < -0.5 + proposed_r
            # Force pushes right: (-0.5 + proposed_r) - x
            left_violation = np.maximum(0, (-0.5 + proposed_r) - P[:, 0])
            forces[:, 0] += left_violation
            
            # Right wall: x > 0.5 - proposed_r
            right_violation = np.maximum(0, P[:, 0] - (0.5 - proposed_r))
            forces[:, 0] -= right_violation
            
            # Bottom wall: y < -0.5 + proposed_r
            bottom_violation = np.maximum(0, (-0.5 + proposed_r) - P[:, 1])
            forces[:, 1] += bottom_violation
            
            # Top wall: y > 0.5 - proposed_r
            top_violation = np.maximum(0, P[:, 1] - (0.5 - proposed_r))
            forces[:, 1] -= top_violation
            
            # Update positions
            # Add small noise to escape local minima
            noise = np.random.randn(N, 2) * noise_scale
            P += alpha * (forces + noise)
            
            # Hard clip to square to ensure validity at all times
            # The radius constraint is handled by forces, but we must stay in square [-0.5, 0.5]
            # Actually, if we clip to [-0.5, 0.5], we might violate the proposed_r boundary constraint
            # if proposed_r > 0. But we are increasing r slowly.
            # If P is clipped, the wall force will be 0 in next iteration if it's exactly at boundary?
            # No, if P is at -0.5, and proposed_r > 0, then (-0.5 + proposed_r) - (-0.5) = proposed_r > 0.
            # So force is still pushing in.
            P[:, 0] = np.clip(P[:, 0], -0.5, 0.5)
            P[:, 1] = np.clip(P[:, 1], -0.5, 0.5)
            
        # Check if proposed_r is feasible
        actual_r = get_min_radius(P)
        
        if actual_r >= proposed_r - 1e-9:
            # Success
            target_r = proposed_r
            success_count += 1
            # Increase epsilon slightly to push harder
            if epsilon < 0.001:
                epsilon *= 1.05
            # Reduce noise as we get closer to optimum?
            # noise_scale = max(0.0001, noise_scale * 0.99)
        else:
            # Fail
            fail_count += 1
            # Decrease epsilon
            epsilon = max(1e-6, epsilon * 0.9)
            # Add a larger kick to escape local minimum
            P += np.random.randn(N, 2) * 0.01
            P[:, 0] = np.clip(P[:, 0], -0.5, 0.5)
            P[:, 1] = np.clip(P[:, 1], -0.5, 0.5)
            
            # If we fail many times, reset to current best?
            if fail_count > 50:
                # Re-center or re-jitter?
                # For now, just continue with smaller steps
                fail_count = 0
                epsilon = 1e-6
                noise_scale = 0.005

    return P
