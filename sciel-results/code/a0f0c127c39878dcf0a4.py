"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Initial random sampling to estimate the linear model parameters.
    2. Use the estimated gradient to take large, informed steps towards the minimizer.
    3. Refine with local (1+1)-ES.
    4. Deadline guard to ensure we finish within 300s (with margin).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Phase 1: Model Estimation
        # We need at least dim + 1 points to solve for a linear model f(x) = c + a^T x.
        # Let's use 10 points to be robust.
        n_model = 10
        X = np.zeros((n_model, dim))
        F = np.zeros(n_model)
        
        # Centered sampling for better conditioning
        for i in range(n_model):
            x = rng.uniform(-4, 4, dim)
            X[i] = x
            F[i] = func(x)
            
        # Solve for linear model: F = X @ a + c
        # Augmented matrix
        X_aug = np.hstack([X, np.ones((n_model, 1))])
        try:
            params, residuals, rank, sv = np.linalg.lstsq(X_aug, F, rcond=None)
            a = params[:dim]
            c = params[dim]
        except:
            # Fallback to random if lstsq fails
            a = np.zeros(dim)
            c = 0.0

        budget_used = n_model
        best_f = np.min(F)
        best_x = X[np.argmin(F)].copy()

        # Phase 2: Gradient Descent / Informed Steps
        # The gradient of the linear model is 'a'.
        # We want to minimize, so we move in direction -a.
        # We need to estimate a good step size.
        # The minimizer of the linear model over the box is at the boundary opposite to 'a'.
        # However, the true function might not be perfectly linear everywhere, but f5 is "Linear slope".
        # Let's try to jump towards the theoretical minimum of the linear model within the box.
        
        # Theoretical minimum of c + a^T x on [-5, 5]^dim:
        # x_i = -5 * sign(a_i)
        x_theoretical = -5.0 * np.sign(a)
        # Check if this point is within bounds (it is, by construction)
        # But maybe the linear model is only valid locally.
        # Let's evaluate it.
        f_theoretical = func(x_theoretical)
        budget_used += 1
        
        if f_theoretical < best_f:
            best_f = f_theoretical
            best_x = x_theoretical.copy()
        else:
            # If the jump was bad, maybe the model is poor or we are far from the linear region.
            # Let's try a step in the gradient direction from the current best.
            # Normalize gradient
            norm_a = np.linalg.norm(a)
            if norm_a > 1e-8:
                grad_dir = -a / norm_a
                # Try a few step sizes
                for step_size in [4.0, 2.0, 1.0, 0.5]:
                    if time.time() > deadline:
                        break
                    x_trial = np.clip(best_x + step_size * grad_dir, -5, 5)
                    f_trial = func(x_trial)
                    budget_used += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        # If this worked, maybe we can do another step?
                        # But let's be conservative and switch to ES refinement.
                        break

        # Phase 3: Local Refinement with (1+1)-ES
        # Start from best_x found so far
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        # Estimate a good initial sigma based on the gradient magnitude?
        # Or just start with a moderate sigma.
        # Since we might have jumped far, sigma=1.0 is reasonable for dim=5.
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
