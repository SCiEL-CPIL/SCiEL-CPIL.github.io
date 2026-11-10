<<<<<<< SEARCH
    def gradient_optimize(points, max_iter=200):
        """Gradient-based optimization using subgradients of the min distance."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.005
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Build KDTree for fast neighbor queries
            tree = cKDTree(points)
            
            # Compute subgradient for each point
            grads = np.zeros_like(points)
            
            for i in range(N):
                r_i = np.linalg.norm(points[i])
                
                # Boundary force: push inward if close to boundary
                if r_i > 1.0 - current_score * 1.5:
                    if r_i > 1e-10:
                        grads[i] += -0.5 * points[i] / r_i
                    else:
                        # At center, push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        grads[i] += 0.5 * np.array([math.cos(angle), math.sin(angle)])
                
                # Find close neighbors
                neighbors = tree.query_ball_point(points[i], 2.0 * current_score * 1.5)
                
                for j in neighbors:
                    if j == i:
                        continue
                    d_vec = points[i] - points[j]
                    d = np.linalg.norm(d_vec)
                    if d > 1e-10:
                        # If too close, push apart
                        if d < 2.0 * current_score:
                            strength = (2.0 * current_score - d) / (2.0 * current_score)
                            grads[i] += strength * d_vec / d
                        elif d < 2.0 * current_score * 1.5:
                            # Mild repulsion for near-miss
                            strength = 0.1 * (2.0 * current_score * 1.5 - d) / (2.0 * current_score * 1.5)
                            grads[i] += strength * d_vec / d
            
            # Normalize gradients
            grad_norms = np.linalg.norm(grads, axis=1)
            max_grad = np.max(grad_norms)
            if max_grad > 1e-10:
                grads /= max_grad
            
            # Update points
            points += lr * grads
            
            # Project back to unit circle
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 0.999
            if np.any(mask):
                points[mask] *= 0.999 / radii[mask]
            
            # Compute score
            new_score = compute_score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                lr *= 0.95
            
            if no_improve > 20 or lr < 1e-6:
                break
        
        return best_points, best_score
=======
    def gradient_optimize(points, max_iter=200):
        """
        Vectorized gradient-based optimization using a smooth surrogate objective.
        Uses full pairwise distances for N=300 which is fast in NumPy.
        """
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve = 0
        softness = 0.1  # Softness parameter for the surrogate
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Compute pairwise distances and vectors
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Target distance is 2 * current_score
            target_d = 2.0 * current_score
            
            # Smooth repulsive force
            # Force is strong if d < target_d, decays smoothly otherwise
            # Using a soft-min-like approach: 
            # penalty = max(0, target_d - d) is not smooth.
            # Use: f(d) = (target_d - d) / (softness + |target_d - d|) if d < target_d + softness else 0
            # Or simpler: just apply force if d < target_d * 1.1, scaled by (target_d - d)
            
            # We want to maximize min(d_ij/2, 1-r_i)
            # Gradient of min is sum of gradients of active constraints.
            # Active constraints: d_ij < 2*r_target OR 1-r_i < r_target
            
            grads = np.zeros_like(points)
            
            # 1. Boundary forces
            radii = np.linalg.norm(points, axis=1)
            # If close to boundary, push inward
            # Smooth penalty: if r_i > 1 - r_target, push in
            boundary_mask = radii > (1.0 - current_score)
            if np.any(boundary_mask):
                # Force direction is -points[i] / r_i
                # Magnitude proportional to violation
                violations = radii[boundary_mask] - (1.0 - current_score)
                # Normalize and scale
                # We want to move inward
                inward_vecs = -points[boundary_mask] / radii[boundary_mask, np.newaxis]
                # Scale by violation magnitude, but keep it smooth
                # Using a simple linear scale
                grads[boundary_mask] += inward_vecs * (violations / current_score)
            
            # 2. Pairwise repulsive forces
            # Only consider pairs where d < target_d * 1.2 to save computation? 
            # For N=300, full matrix is fine.
            
            # We want to push i away from j if d_ij < target_d
            # Gradient of d_ij w.r.t p_i is (p_i - p_j) / d_ij
            # We want to increase d_ij, so force is + (p_i - p_j) / d_ij * weight
            
            # Weight: high if d < target_d, low otherwise.
            # Let's use a smooth step function for the weight.
            # weight = 1 if d < target_d, 0 if d > target_d + eps
            # Smoothly: weight = sigmoid((target_d - d) / softness)
            
            # To avoid division by zero, mask diagonal
            dists_safe = np.where(dists < 1e-10, 1e-10, dists)
            
            # Calculate forces
            # Force on i from j: (p_i - p_j) / d_ij * weight_ij
            # weight_ij = 1 / (1 + exp((d_ij - target_d) / softness))
            # This is 0.5 at d=target_d, ~1 for d<<target_d, ~0 for d>>target_d
            
            # However, we only want to act if d is *less* than target_d significantly?
            # Actually, if d > target_d, we don't need to push apart.
            # But if d is slightly larger, pushing apart might help if other constraints tighten?
            # No, we want to maximize the *minimum*. So we only care about the active constraints (the minimums).
            # So we should only apply force if d_ij is close to the current minimum distance.
            
            # Let's identify the "active" pairs: those with d_ij < 2 * current_score * 1.05
            active_mask = dists < (2.0 * current_score * 1.05)
            # Exclude diagonal
            np.fill_diagonal(active_mask, False)
            
            if np.any(active_mask):
                # Get active pairs
                # We can vectorize the force calculation
                # Force on i = sum_j (p_i - p_j) / d_ij * w_ij
                # Let's compute the vector sum directly
                
                # Normalize diff vectors
                # diff[i,j] = p_i - p_j
                # unit_vec[i,j] = diff[i,j] / d_ij
                unit_vecs = diff / dists_safe[:, :, np.newaxis]
                
                # Weights
                # Only apply strong weight if d < target_d
                # w = 1 if d < target_d else 0 (smoothed)
                # Let's use a linear ramp: w = max(0, 1 - (d - target_d)/softness) if d > target_d? No.
                # We want w=1 for d <= target_d, w=0 for d >= target_d + softness.
                w = np.clip(1.0 - (dists - target_d) / softness, 0.0, 1.0)
                # Zero out inactive
                w[~active_mask] = 0.0
                
                # Force on i is sum over j of w[i,j] * unit_vec[i,j]
                # Note: unit_vec[i,j] points from j to i. So adding it pushes i away from j. Correct.
                force_sum = np.sum(w[:, :, np.newaxis] * unit_vecs, axis=1)
                
                # Scale the force
                # The magnitude of force_sum depends on number of neighbors.
                # Normalize? Or just scale by lr.
                grads += force_sum
            
            # Normalize gradients to prevent large steps
            grad_norms = np.linalg.norm(grads, axis=1)
            max_grad = np.max(grad_norms)
            if max_grad > 1e-10:
                grads /= max_grad
            
            # Update points
            points += lr * grads
            
            # Project back to unit circle
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 0.999
            if np.any(mask):
                points[mask] *= 0.999 / radii[mask]
            
            # Compute score
            new_score = compute_score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
                no_improve = 0
                # Increase lr slightly if improving? Or keep steady.
                # lr = min(lr * 1.05, 0.05)
            else:
                no_improve += 1
                lr *= 0.9
                # If stuck, try a small random jitter to escape local minima
                if no_improve > 10:
                    points += np.random.randn(N, 2) * 0.001
                    points = np.clip(points, -0.999, 0.999) # Rough clip, projection will fix
            
            if no_improve > 50 or lr < 1e-6:
                break
        
        return best_points, best_score
>>>>>>> REPLACE

