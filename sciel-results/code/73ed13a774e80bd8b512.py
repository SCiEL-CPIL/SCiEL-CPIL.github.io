"""Model-and-refine optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate the linear model f(x) ≈ c + a·x using forward differences.
    2. Use the estimated gradient to take large steps toward the minimizer.
    3. Refine with adaptive local search.
    4. Deadline guard to ensure we finish within the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin (300s limit)
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Estimate linear model using forward differences
        # f(x) ≈ c + a^T * x
        # We need dim+1 evaluations to estimate a and c
        # Use a point near the center to avoid boundary issues
        
        x0 = np.zeros(dim)
        fx0 = func(x0)
        
        # Forward differences to estimate gradient
        eps = 0.1
        grad = np.zeros(dim)
        calls = 1  # fx0
        
        for i in range(dim):
            ei = np.zeros(dim)
            ei[i] = eps
            xi = np.clip(x0 + ei, -5, 5)
            fxi = func(xi)
            calls += 1
            grad[i] = (fxi - fx0) / eps
        
        # Estimate c and a from the linear model
        # f(x) ≈ fx0 + grad^T * (x - x0)
        # The minimizer of the linear model is where the gradient points
        # If grad is all zeros, we're at a minimum (unlikely for linear slope)
        
        # Phase 2: Take steps toward the minimizer
        # For a linear function f(x) = c + a^T x, the minimizer is at the boundary
        # where a^T x is minimized. Since x ∈ [-5, 5]^d, the minimizer is at
        # x_i = -5 * sign(a_i) for each i (if a_i ≠ 0)
        
        # But we don't know the true a, we have an estimate. Let's use it.
        # Actually, for linear slope, the function is f(x) = c + sum_i a_i * x_i
        # The minimum over [-5,5]^d is at x_i = -5*sign(a_i)
        
        # Let's try to move toward that point
        # But we should verify with actual function evaluations
        
        # Phase 2: Adaptive search
        x = x0.copy()
        fx = fx0
        best_x = x.copy()
        best_fx = fx
        
        # Check if we're near the boundary already
        # Try moving in the negative gradient direction
        if np.linalg.norm(grad) > 1e-8:
            # Move toward the estimated minimizer
            step_size = 10.0  # Large step to reach the boundary
            x_new = np.clip(x - step_size * grad / np.linalg.norm(grad), -5, 5)
            fx_new = func(x_new)
            calls += 1
            
            if fx_new < best_fx:
                best_x = x_new.copy()
                best_fx = fx_new
                x = x_new.copy()
                fx = fx_new
        
        # Phase 3: Local refinement using coordinate descent
        # Use remaining budget for refinement
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_fx
        
        # Check deadline
        if time.time() > deadline:
            return best_x, best_fx
        
        # Adaptive step size for refinement
        sigma = 0.5
        fail_count = 0
        max_fails = 5
        
        for _ in range(remaining_budget):
            if time.time() > deadline:
                break
            
            # Try a random perturbation
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                sigma *= 1.2
                fail_count = 0
            else:
                sigma *= 0.7
                fail_count += 1
                if fail_count >= max_fails:
                    # Restart with larger sigma or different direction
                    sigma = max(0.1, sigma * 2)
                    fail_count = 0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
