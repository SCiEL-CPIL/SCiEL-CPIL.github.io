"""Multi-start local optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315.0 
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Initial Exploration
        # Sample 10 points to find a good starting basin
        n_init = 10
        for _ in range(n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            wrapped_func(x)

        # Phase 2: Multi-start Local Refinement
        # Use L-BFGS-B which is efficient for smooth functions like BBOB f22
        # We will run multiple restarts, each starting from the current best
        # or a perturbation of it, to escape local minima.
        
        # L-BFGS-B parameters
        bounds = [(-5.0, 5.0)] * dim
        
        # Strategy:
        # 1. Run L-BFGS-B from best_x.
        # 2. If it improves, restart from new best.
        # 3. If it stagnates or fails, try a perturbed start.
        # Limit iterations to ensure we don't consume all budget in one run.
        
        while eval_count < budget and time.time() < deadline:
            # Determine starting point
            # 50% chance to restart from best, 50% chance to perturb best
            # This balances exploitation and exploration
            if rng.random() < 0.5:
                x_start = best_x.copy()
            else:
                # Perturb with a scale that decreases as we get closer to convergence?
                # For now, fixed small perturbation
                scale = 0.5
                x_start = best_x + rng.normal(0, scale, dim)
                x_start = np.clip(x_start, -5.0, 5.0)

            # Run L-BFGS-B
            # maxiter=50 is usually enough for 5D smooth functions to converge
            # or hit the tolerance
            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Update best if improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If minimization fails, just continue loop
                pass

            # Check if we should break early if no progress
            # (Optional: track progress, but for now rely on budget/time)
            
        # Phase 3: Final Polish
        # If we have budget left, do a few more random samples near best
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.1, dim)
            x = np.clip(x, -5.0, 5.0)
            wrapped_func(x)

        return best_x, best_f

# EVOLVE-BLOCK-END
