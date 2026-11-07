"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quadratic Surrogate Model Optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.buffer_size = 25  # Number of points to keep for fitting

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Start time for deadline check
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        # Initialize with a few random points to establish a baseline
        # The Bent Cigar is globally unimodal, so random start is okay, 
        # but we want to be in the basin.
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        # Store history of (x, f(x))
        history_x = [x.copy()]
        history_f = [fx]
        
        # Initial exploration: take a few random steps to get a sense of scale
        # and populate the buffer with diverse points.
        # We use a simple random walk for the first few steps to fill the buffer.
        sigma_init = 2.0
        for i in range(min(10, budget - 1)):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma_init * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
            history_x.append(y.copy())
            history_f.append(fy)
            # Keep buffer size limited
            if len(history_x) > self.buffer_size:
                history_x.pop(0)
                history_f.pop(0)

        calls_made = len(history_x)
        
        # Main loop: Surrogate-based optimization
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Fit quadratic model to the last 'buffer_size' points
            # We need at least dim+1 points, but for a full quadratic (dim*(dim+1)/2 + dim + 1 params), 
            # we ideally want more. Let's use the last min(len, 2*dim+10) points.
            n_pts = len(history_x)
            if n_pts < dim + 1:
                # Fallback to random walk if not enough points
                y = np.clip(x + 0.5 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                history_x.append(y.copy())
                history_f.append(fy)
                if len(history_x) > self.buffer_size:
                    history_x.pop(0)
                    history_f.pop(0)
                calls_made += 1
                continue

            # Select points for fitting. Use the most recent ones.
            # If we have more than buffer_size, we already trimmed it.
            X = np.array(history_x)
            F = np.array(history_f)
            
            # Center and scale for numerical stability
            mean_x = X.mean(axis=0)
            # Estimate scale from spread
            std_x = X.std(axis=0)
            std_x[std_x < 1e-6] = 1.0
            X_scaled = (X - mean_x) / std_x
            
            # Build design matrix for quadratic fit
            # f(x) = c + g^T x + x^T Q x
            # We fit: F = [1, X_scaled, X_scaled^2] * theta
            # Features: 1, x_i, x_i^2, x_i x_j (i<j)
            n_features = 1 + dim + dim*(dim+1)//2
            A = np.zeros((n_pts, n_features))
            A[:, 0] = 1.0
            A[:, 1:1+dim] = X_scaled
            # Quadratic terms
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        A[:, 1+dim+i] = X_scaled[:, i] ** 2
                    else:
                        # Index for x_i x_j in the feature vector
                        # We need a consistent mapping.
                        # Let's just use a simple loop to fill the matrix properly.
                        # Actually, constructing A explicitly is safer.
                        pass
            
            # Reconstruct A properly to avoid index confusion
            A = np.zeros((n_pts, n_features))
            A[:, 0] = 1.0
            A[:, 1:1+dim] = X_scaled
            
            # Fill quadratic terms
            quad_start = 1 + dim
            idx = quad_start
            for i in range(dim):
                for j in range(i, dim):
                    A[:, idx] = X_scaled[:, i] * X_scaled[:, j]
                    idx += 1
            
            # Solve least squares
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A, F, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback to random step
                y = np.clip(x + 0.1 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                history_x.append(y.copy())
                history_f.append(fy)
                if len(history_x) > self.buffer_size:
                    history_x.pop(0)
                    history_f.pop(0)
                calls_made += 1
                continue

            # Extract parameters
            c = theta[0]
            g_scaled = theta[1:1+dim]
            # Reconstruct Hessian H from quadratic coefficients
            # f(x) = ... + sum(theta_quad[i,j] * x_i x_j)
            # Hessian H is such that f ~ 0.5 * x^T H x
            # The coefficient of x_i^2 is theta_quad[i,i] = H_ii / 2 ? No.
            # Let's derive:
            # f(x) = c + g^T x + sum_{i<=j} q_{ij} x_i x_j
            # Gradient: g_i + sum_{j>=i} q_{ji} x_j + sum_{j<i} q_{ij} x_j
            # Hessian: H_ii = 2 q_ii, H_ij = q_ij + q_ji
            # In our A matrix, we have terms q_{ij} for i<=j.
            # So theta[quad_start + offset] corresponds to q_{ij}.
            
            H_scaled = np.zeros((dim, dim))
            idx = quad_start
            for i in range(dim):
                for j in range(i, dim):
                    q_ij = theta[idx]
                    H_scaled[i, j] = q_ij
                    H_scaled[j, i] = q_ij
                    if i == j:
                        H_scaled[i, j] = 2.0 * q_ij
                    idx += 1
            
            # The gradient in scaled space is g_scaled.
            # The Hessian in scaled space is H_scaled.
            # We want to minimize f(x) in original space.
            # Let x = mean_x + std_x * z.
            # f(x) ~ c + g_scaled^T z + 0.5 z^T H_scaled z
            # Minimizer z* = - H_scaled^{-1} g_scaled
            # Then x* = mean_x + std_x * z*
            
            # Regularize Hessian to ensure invertibility
            # Add small lambda to diagonal
            lam = 1e-4
            H_reg = H_scaled + lam * np.eye(dim)
            
            try:
                z_star = np.linalg.solve(H_reg, -g_scaled)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse or small step
                z_star = np.linalg.pinv(H_reg) @ (-g_scaled)
                # Or just random
                z_star = rng.standard_normal(dim) * 0.1

            # Map back to original space
            x_proposed = mean_x + z_star * std_x
            
            # Clip to bounds
            x_proposed = np.clip(x_proposed, -5, 5)
            
            # Evaluate
            fy = func(x_proposed)
            calls_made += 1
            
            if fy < fx:
                x, fx = x_proposed, fy
                history_x.append(x_proposed.copy())
                history_f.append(fy)
            else:
                # If the step was bad, we might be far from the basin or the model is wrong.
                # Add the point to history anyway to update the model, but don't move x.
                # Actually, adding a bad point can pollute the model.
                # Strategy: If step fails, take a smaller random step to explore locally.
                # But for now, let's just add the proposed point to history to keep the model updated?
                # No, if it's worse, it's not representative of the minimum.
                # Let's add a small perturbation from current best to history.
                y = np.clip(x + 0.1 * rng.standard_normal(dim), -5, 5)
                fy_retry = func(y)
                calls_made += 1
                if fy_retry < fx:
                    x, fx = y, fy_retry
                history_x.append(y.copy())
                history_f.append(fy_retry)
                
            # Manage buffer
            if len(history_x) > self.buffer_size:
                history_x.pop(0)
                history_f.pop(0)

        return x, fx
# EVOLVE-BLOCK-END
