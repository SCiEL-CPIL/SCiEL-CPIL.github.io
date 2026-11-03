    def smooth_objective_and_grad(x, K=50):
        pts = x.reshape(-1, 2)
        N = len(pts)
        
        # Pairwise distances
        # To save memory, we can compute in chunks or just use full matrix for N=200 (40k entries, fine)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Wall distances
        # dist to left: 0.5 + x
        # dist to right: 0.5 - x
        # dist to bottom: 0.5 + y
        # dist to top: 0.5 - y
        wall_dists = np.concatenate([
            (0.5 + pts[:, 0]).reshape(-1, 1),
            (0.5 - pts[:, 0]).reshape(-1, 1),
            (0.5 + pts[:, 1]).reshape(-1, 1),
            (0.5 - pts[:, 1]).reshape(-1, 1)
        ])
        
        # Combine all distances
        # We need to be careful with indices for gradient
        # Let's compute the log-sum-exp value
        # Z = sum(exp(-K * d))
        # To avoid underflow, we can subtract min_d
        # But for gradient, we need weights w_i = exp(-K * d_i) / Z
        
        all_dists = np.concatenate([dist[dist < np.inf].flatten(), wall_dists.flatten()])
        
        # If any distance is 0, r=0.
        if np.min(all_dists) < 1e-10:
            return 0.0, np.zeros_like(x)
            
        min_d = np.min(all_dists)
        # Shift to prevent underflow
        shifted = -K * (all_dists - min_d)
        exp_shifted = np.exp(shifted)
        Z = np.sum(exp_shifted)
        
        # Objective: - (min_d + (1/K) * log(Z))
        # Note: log-sum-exp approx is min_d + (1/K) log(Z)
        obj_val = -(min_d + (1.0 / K) * np.log(Z))
        
        # Gradient
        # grad = - (1/K) * (1/Z) * sum( -K * w_i_unnormalized * grad_d_i )
        # Wait, d/dx (min_d + 1/K log Z)
        # Let f = min_d + 1/K log(sum exp(-K d_j))
        # df/dx = d(min_d)/dx + 1/K * 1/Z * sum( -K exp(-K d_j) * d(d_j)/dx )
        # df/dx = d(min_d)/dx - 1/Z * sum( exp(-K d_j) * d(d_j)/dx )
        
        # The term d(min_d)/dx is non-differentiable, but the log-sum-exp term is smooth.
        # Actually, the log-sum-exp IS the smooth approximation. We don't add the raw min_d.
        # The approximation is: min(d) ≈ (1/K) log(sum exp(-K d)) + min_d ? No.
        # The standard soft-min is: softmin(d) = -1/K log(sum exp(-K d)).
        # This converges to min(d) as K->inf.
        # So obj = - softmin(d) = 1/K log(sum exp(-K d)).
        
        # Let's use obj = 1/K log(sum exp(-K d))
        # grad obj = 1/K * 1/Z * sum( -K exp(-K d_j) * grad d_j )
        # grad obj = - 1/Z * sum( exp(-K d_j) * grad d_j )
        
        # We need to accumulate gradients for each point.
        grad_x = np.zeros_like(x)
        
        # Weights
        weights = exp_shifted / Z
        
        # We need to map weights back to pairs and walls.
        # This is complex to vectorize easily without explicit loops or careful index mapping.
        
        # Given the complexity and time, let's fall back to a simpler, robust method:
        # **Gradient Descent with Collision Resolution**
        
        # Or, simpler: **L-BFGS-B with non-smooth objective**.
        # L-BFGS-B uses finite difference gradients by default if not provided.
        # It is robust to some non-smoothness.
        # With bounds, it will stay inside.
        
        # Let's try L-BFGS-B with bounds and the original non-smooth objective.
        # It might be slow, but it will produce valid results.
        # To speed it up, we can reduce maxfev.
        
        return obj_val, grad_x # Placeholder
    