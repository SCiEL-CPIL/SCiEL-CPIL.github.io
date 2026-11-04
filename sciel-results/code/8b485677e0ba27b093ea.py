import numpy as np

N = 50

def construct():
    """
    Uses Steepest Ascent on the Minimum Radius to maximize the packing density.
    This approach identifies the "active" constraints (pairs and walls limiting the radius)
    and moves the particles in the direction that increases the minimum distance.
    """
    rng = np.random.RandomState(42)
    
    def compute_radius_and_active(pts, delta=0.001):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Wall distances
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        # Current radius
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Identify active constraints
        # Pairs where dist < 2*(r + delta)
        active_pairs_mask = dists < 2.0 * (r + delta)
        np.fill_diagonal(active_pairs_mask, False)
        
        # Walls where dist < r + delta
        active_walls_mask = np.zeros((N, 4), dtype=bool)
        active_walls_mask[:, 0] = d_left < r + delta   # Left
        active_walls_mask[:, 1] = d_right < r + delta  # Right
        active_walls_mask[:, 2] = d_bottom < r + delta # Bottom
        active_walls_mask[:, 3] = d_top < r + delta    # Top
        
        return r, active_pairs_mask, active_walls_mask, dists, diff, d_left, d_right, d_bottom, d_top

    def run_steepest_ascent(initial_points, steps=5000):
        points = initial_points.copy()
        best_r = 0.0
        best_points = points.copy()
        
        # Initial radius
        r, _, _, _, _, _, _, _, _ = compute_radius_and_active(points)
        best_r = r
        best_points = points.copy()
        
        step_size = 0.01
        no_improve_count = 0
        
        for step in range(steps):
            r, active_pairs_mask, active_walls_mask, dists, diff, d_left, d_right, d_bottom, d_top = compute_radius_and_active(points)
            
            # Compute gradient of the minimum radius
            # We approximate the gradient of min(dist) by the average of the gradients of the active constraints.
            grad = np.zeros_like(points)
            
            # Pairwise contributions
            # For each active pair (i, j), the gradient of dist_ij w.r.t p_i is (p_i - p_j) / dist_ij
            # We want to maximize the minimum distance, so we move in the direction of the gradient.
            # However, the objective is min(dist). The subgradient is a convex combination of the gradients of the active pairs.
            # A simple heuristic: sum the unit vectors for active pairs.
            
            # Efficiently compute gradients for active pairs
            # active_pairs_mask is symmetric. We only need to consider i < j to avoid double counting, 
            # but the gradient for i is +u and for j is -u.
            # Let's iterate over active pairs.
            
            # To be faster, we can use the full matrix.
            # grad_i = sum_j (diff[i,j] / dists[i,j]) * active_pairs_mask[i,j]
            # But dists[i,i] is 0. We handle that.
            
            # Normalize diff by dists, handling zeros
            safe_dists = np.where(dists > 1e-8, dists, 1.0)
            unit_diff = diff / safe_dists[:, :, np.newaxis]
            unit_diff[dists < 1e-8] = 0.0
            
            # Sum unit vectors for active pairs
            # grad[i] = sum_j unit_diff[i,j] * active_pairs_mask[i,j]
            pair_grad = np.sum(unit_diff * active_pairs_mask[:, :, np.newaxis], axis=1)
            
            # Wall contributions
            # For wall 0 (Left, x < -0.5 + r), gradient is (1, 0)
            # For wall 1 (Right, x > 0.5 - r), gradient is (-1, 0)
            # For wall 2 (Bottom, y < -0.5 + r), gradient is (0, 1)
            # For wall 3 (Top, y > 0.5 - r), gradient is (0, -1)
            
            wall_grads = np.zeros_like(points)
            wall_grads[:, 0] += active_walls_mask[:, 0] * 1.0
            wall_grads[:, 0] -= active_walls_mask[:, 1] * 1.0
            wall_grads[:, 1] += active_walls_mask[:, 2] * 1.0
            wall_grads[:, 1] -= active_walls_mask[:, 3] * 1.0
            
            total_grad = pair_grad + wall_grads
            
            # Normalize gradient to have unit length (or scale it)
            grad_norm = np.linalg.norm(total_grad, axis=1, keepdims=True)
            grad_norm = np.where(grad_norm > 1e-8, grad_norm, 1.0)
            direction = total_grad / grad_norm
            
            # Update positions
            new_points = points + step_size * direction
            
            # Clamp to square
            new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
            new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
            
            # Evaluate new radius
            new_r, _, _, _, _, _, _, _, _ = compute_radius_and_active(new_points)
            
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                # Optionally increase step size slightly if improving
                # step_size = min(step_size * 1.05, 0.02)
            else:
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.95
                # If stuck for too long, try a random perturbation (Simulated Annealing style)
                if no_improve_count > 50:
                    # Accept worse solution with some probability (Metropolis criterion)
                    # Or just restart from best with perturbation
                    # Let's try a small random kick
                    if rng.rand() < 0.1:
                        points = best_points + rng.normal(0, 0.01, size=best_points.shape)
                        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
                        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
                        step_size = 0.01
                        no_improve_count = 0
                    else:
                        # Decay step size more aggressively
                        step_size *= 0.9
                        if step_size < 1e-6:
                            break

        return best_points, best_r

    # Generate multiple initial configurations
    best_points_overall = None
    best_r_overall = 0.0
    
    for m in [7, 8, 5, 10]:
        for n in [7, 8, 5, 10]:
            if m * n < N:
                continue
            if m * n > 120: # Skip too dense grids
                continue
            ticks_x = (np.arange(m) + 0.5) / m - 0.5
            ticks_y = (np.arange(n) + 0.5) / n - 0.5
            grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
            points = grid[:N].astype(float)
            # Add small jitter to break symmetry
            points += rng.uniform(-0.005, 0.005, size=points.shape)
            points = np.clip(points, -0.49, 0.49)
            
            pts, r = run_steepest_ascent(points, steps=3000)
            if r > best_r_overall:
                best_r_overall = r
                best_points_overall = pts

    return best_points_overall
