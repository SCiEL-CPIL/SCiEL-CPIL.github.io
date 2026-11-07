"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using Radial Sweep initialization 
    followed by Gradient Descent with Backtracking Line Search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 275  # Safety margin before 300s limit
        calls_made = 0
        x_best = np.zeros(dim)
        f_best = np.inf

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def update_best(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # --- Phase 1: Radial Sweep Initialization ---
        # Evaluate origin
        f0 = func(x_best)
        calls_made += 1
        update_best(x_best, f0)

        # Generate radial sweep points
        # We sample directions on the sphere and radii
        # For dim=5, we can use a quasi-random set or grid.
        # Let's use a deterministic set of directions.
        # We want to cover the space [-5, 5]^5.
        
        # Simple grid of directions? Too many.
        # Let's use a set of vectors: axes, diagonals, and some random-ish deterministic ones.
        directions = []
        # Unit vectors
        for i in range(dim):
            v = np.zeros(dim)
            v[i] = 1.0
            directions.append(v)
            v[i] = -1.0
            directions.append(v)
        # Diagonals
        v_diag = np.ones(dim)
        directions.append(v_diag)
        v_diag_neg = -v_diag
        directions.append(v_diag_neg)
        
        # Add some "random" deterministic directions based on index
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = rng.normal(size=dim)
            norm = np.linalg.norm(v)
            if norm > 0:
                directions.append(v / norm)

        radii = [1.0, 2.0, 3.0, 4.0]

        for d in directions:
            for r in radii:
                if calls_made >= budget - 100 or time.time() >= deadline:
                    break
                x_trial = clip(d * r)
                f_trial = func(x_trial)
                calls_made += 1
                update_best(x_trial, f_trial)
            if calls_made >= budget - 100 or time.time() >= deadline:
                break

        # --- Phase 2: Gradient Descent with Backtracking Line Search ---
        # Use x_best as starting point
        x_curr = x_best.copy()
        f_curr = f_best
        
        # Initial step size
        alpha = 0.1
        # Learning rate for step size adjustment
        alpha_min = 1e-6
        alpha_max = 1.0
        alpha_factor = 0.5
        
        # Finite difference step for gradient
        h = 0.01
        
        max_iter = 200
        for i in range(max_iter):
            if calls_made >= budget - 10 or time.time() >= deadline:
                break
            
            # Estimate gradient using forward differences
            grad = np.zeros(dim)
            for j in range(dim):
                if calls_made >= budget - 10 or time.time() >= deadline:
                    break
                x_p = x_curr.copy()
                x_p[j] += h
                x_p = clip(x_p)
                f_p = func(x_p)
                calls_made += 1
                # If we hit the boundary, the gradient estimate is biased, but it's okay for now.
                # Ideally use central diff, but it costs 2x.
                # Let's stick to forward diff for efficiency.
                grad[j] = (f_p - f_curr) / h
            
            if np.linalg.norm(grad) < 1e-8:
                break
                
            # Backtracking Line Search
            # Condition: f(x - alpha * grad) < f_curr - c * alpha * ||grad||^2
            c = 1e-4
            improved = False
            alpha_trial = alpha
            
            for ls_iter in range(10):
                if calls_made >= budget - 10 or time.time() >= deadline:
                    break
                
                x_trial = clip(x_curr - alpha_trial * grad)
                f_trial = func(x_trial)
                calls_made += 1
                
                # Armijo condition
                if f_trial < f_curr - c * alpha_trial * np.dot(grad, grad):
                    x_curr = x_trial
                    f_curr = f_trial
                    update_best(x_curr, f_curr)
                    improved = True
                    # Increase step size for next iteration if we succeeded with a large step?
                    # Or keep it? Let's keep it or slightly increase if it was small.
                    if alpha_trial < alpha_max:
                        alpha = min(alpha_max, alpha_trial * 1.5)
                    break
                else:
                    alpha_trial *= alpha_factor
            
            if not improved:
                # If no step size worked, reduce alpha significantly
                alpha *= 0.1
                if alpha < alpha_min:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
