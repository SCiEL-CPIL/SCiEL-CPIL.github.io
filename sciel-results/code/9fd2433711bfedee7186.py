"""Rewritten program with a quadratic surrogate model for the Bent Cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Optimizer:
    """
    Optimizer using a quadratic surrogate model (local paraboloid fit) 
    to navigate the narrow valley of the Bent Cigar function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        d = self.dim
        n = self.budget
        
        # Use a fixed seed for reproducibility in evaluation, though instances are shifted/rotated
        rng = np.random.default_rng(42)

        # Initialize best point
        # The minimizer is in [-4, 4]^5. Start at 0 or random within this box.
        x_best = rng.uniform(-2, 2, d)
        f_best = func(x_best)
        
        # Pool of points for the surrogate model
        # We want a diverse set of points to fit a quadratic.
        # Size K should be at least number of parameters: 1 + d + d*(d+1)/2 = 1 + 5 + 15 = 21.
        # Let's use K=25 to have some redundancy.
        K = 25
        pool = deque(maxlen=K)
        pool.append((x_best.copy(), f_best))
        
        # Current working point for the model step
        x_curr = x_best.copy()
        
        calls_used = 1
        no_improve_count = 0
        max_no_improve = 10 # Restart logic if stuck

        for step in range(1, n):
            if time.time() > deadline:
                break
            
            # 1. Update the surrogate model if we have enough points
            # We need at least d+1 points to estimate gradient, but for Hessian we need more.
            # Let's fit once we have at least d+2 points, but preferably K.
            if len(pool) >= d + 2:
                X = np.array([p[0] for p in pool])
                Y = np.array([p[1] for p in pool])
                
                # Center the data to improve numerical stability of the quadratic fit
                mean_x = X.mean(axis=0)
                X_centered = X - mean_x
                Y_centered = Y - Y.mean() # Not strictly necessary for fitting, but helps with intercept
                
                # Construct feature matrix for quadratic model:
                # f(x) = b0 + b^T x + 0.5 * x^T H x
                # In terms of centered variables z = x - mean_x:
                # f(z + mean) = c + g^T z + 0.5 z^T H z
                # We fit c, g, and H.
                
                # Features: 1, z, z_i*z_j (i<=j)
                # Number of params: 1 + d + d*(d+1)/2
                n_params = 1 + d + d * (d + 1) // 2
                if len(pool) >= n_params:
                    A = np.zeros((len(X), n_params))
                    A[:, 0] = 1.0
                    A[:, 1:1+d] = X_centered
                    # Fill in quadratic terms
                    idx = 1 + d
                    for i in range(d):
                        for j in range(i, d):
                            A[:, idx] = X_centered[:, i] * X_centered[:, j]
                            idx += 1
                    
                    # Solve least squares
                    try:
                        coeffs, residuals, rank, sv = np.linalg.lstsq(A, Y, rcond=None)
                        if rank == n_params: # Full rank, good model
                            c = coeffs[0]
                            g = coeffs[1:1+d]
                            # Reconstruct H from coeffs
                            H = np.zeros((d, d))
                            idx = 1 + d
                            for i in range(d):
                                for j in range(i, d):
                                    val = coeffs[idx]
                                    H[i, j] = val
                                    if i != j:
                                        H[j, i] = val
                                    idx += 1
                            
                            # Regularize Hessian to ensure positive definite (convex) for minimization
                            # Add small identity to avoid singularity
                            H_reg = H + 1e-4 * np.eye(d)
                            
                            # Calculate next step from x_curr
                            # Newton step: x_new = x_curr - H^{-1} * grad_model(x_curr)
                            # Grad model at x_curr (in original coords):
                            # grad_model(x) = g + H * (x - mean_x)
                            z_curr = x_curr - mean_x
                            grad_model = g + H_reg @ z_curr
                            
                            # Solve H_reg * delta = grad_model
                            try:
                                delta = np.linalg.solve(H_reg, grad_model)
                                # Limit step size to avoid jumping out of basin
                                # Max step norm relative to current sigma or box size
                                max_step = 2.0
                                norm_delta = np.linalg.norm(delta)
                                if norm_delta > max_step:
                                    delta = delta / norm_delta * max_step
                                
                                x_new = x_curr + delta
                                x_new = np.clip(x_new, -5, 5)
                                
                            except np.linalg.LinAlgError:
                                # Fallback to gradient descent if Hessian is singular
                                norm_g = np.linalg.norm(grad_model)
                                if norm_g > 1e-12:
                                    step_size = 0.5
                                    x_new = x_curr - step_size * grad_model
                                    x_new = np.clip(x_new, -5, 5)
                                else:
                                    # Gradient is zero, model is flat? Use random step
                                    x_new = x_curr + 0.1 * rng.standard_normal(d)
                                    x_new = np.clip(x_new, -5, 5)
                        else:
                            # Model not full rank, use random exploration
                            x_new = x_curr + 1.0 * rng.standard_normal(d)
                            x_new = np.clip(x_new, -5, 5)
                    except Exception:
                        x_new = x_curr + 1.0 * rng.standard_normal(d)
                        x_new = np.clip(x_new, -5, 5)
                else:
                    # Not enough points for full quadratic, use random walk
                    x_new = x_curr + 1.0 * rng.standard_normal(d)
                    x_new = np.clip(x_new, -5, 5)
            else:
                # Initial phase: random exploration to populate pool
                x_new = x_curr + 1.5 * rng.standard_normal(d)
                x_new = np.clip(x_new, -5, 5)

            # Evaluate
            f_new = func(x_new)
            calls_used += 1
            
            # Update pool
            # We want to keep the best points. 
            # Simple strategy: if new point is better than worst in pool, replace it.
            # Or just append (deque handles size). 
            # For quadratic fitting, diversity is important. 
            # Let's just append. The deque will drop the oldest.
            # However, keeping the *best* points is better for convergence.
            # Let's maintain a list of the best K points.
            # To keep it simple and fast, we'll just append to the deque.
            # The "pool" will contain the most recent K points.
            # This might not be ideal if we move far away, but the model is local.
            pool.append((x_new.copy(), f_new))
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                x_curr = x_new.copy() # Move current position to the new best
                no_improve_count = 0
            else:
                # If we didn't improve, move current position slightly towards new point?
                # Or just stay? 
                # In (1+1) ES, we move if improved, stay if not.
                # Here, we use the model. If the model step didn't improve, 
                # we might be in a non-convex region or model is bad.
                # Let's move x_curr to x_new anyway to keep the local pool relevant?
                # No, if it's worse, we should probably not commit to it.
                # But we need to explore. 
                # Let's keep x_curr as is, but the pool has the new point.
                # Next model fit will include this worse point.
                # To encourage exploration, if no improvement for a while, increase step size.
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Perturb current position to escape local trap
                    x_curr = x_curr + 2.0 * rng.standard_normal(d)
                    x_curr = np.clip(x_curr, -5, 5)
                    no_improve_count = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
