"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Weighted Sum Hill Climbing optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Strategy: Multi-start local search
        # We divide the budget into N restarts, each with K evaluations.
        # N=20, K=10 -> 200 evaluations.
        n_restarts = 20
        steps_per_restart = budget // n_restarts
        if steps_per_restart < 2:
            steps_per_restart = 2
            n_restarts = budget // steps_per_restart

        all_points = []
        all_f = []

        for r in range(n_restarts):
            # Define weight vector to target a specific part of the Pareto front
            # ZDT2 front is convex. We use linear weights.
            # To cover the front, we vary the weight angle.
            # w = [cos(theta), sin(theta)]
            theta = np.pi / 4 * (r / (n_restarts - 1) if n_restarts > 1 else 0.5)
            w = np.array([np.cos(theta), np.sin(theta)])

            # Initial point
            # x1 is the main driver of f1. Sample uniformly.
            # x2..xn should be small for optimality.
            x = np.zeros(n_dim)
            x[0] = rng.uniform(lo[0], hi[0])
            for j in range(1, n_dim):
                # Bias towards 0, but allow some exploration
                # Use a small range for other variables
                x[j] = rng.uniform(lo[j], lo[j] + 0.1 * (hi[j] - lo[j]))
            
            x = np.clip(x, lo, hi)
            f = problem(x)
            used = 1
            
            best_x = x.copy()
            best_f = f.copy()
            best_scalar = np.dot(w, f)

            # Local Hill Climbing
            for step in range(steps_per_restart - 1):
                if used >= budget:
                    break
                
                # Generate a perturbation
                # Use a step size that decreases or is fixed
                # A fixed small step is robust
                scale = 0.1
                delta = rng.normal(0, scale, size=n_dim)
                x_new = best_x + delta
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                scalar_new = np.dot(w, f_new)
                
                if scalar_new < best_scalar:
                    best_x = x_new
                    best_f = f_new
                    best_scalar = scalar_new
            
            # Record the best point found in this restart
            all_points.append(best_x)
            all_f.append(best_f)

        return np.array(all_points)

# EVOLVE-BLOCK-END
