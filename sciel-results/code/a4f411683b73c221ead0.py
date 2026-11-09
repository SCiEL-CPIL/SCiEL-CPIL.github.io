"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time guard for Rosenbrock on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        # Track best value and count evaluations
        best_x = None
        best_f = np.inf
        eval_count = 0
        budget_left = self.budget

        # We need to track evaluations carefully
        # The func is called by us and by scipy
        # We'll wrap func to count and check budget

        global_eval_count = [0]

        def counting_func(x):
            nonlocal budget_left, best_x, best_f
            if global_eval_count[0] >= budget_left:
                # Return a large value to stop optimization
                return 1e10
            global_eval_count[0] += 1
            budget_left -= 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Multi-start strategy:
        # 1. Center (0,0,0,0,0)
        # 2. A few random starts
        # 3. Use L-BFGS-B from each start

        rng = np.random.default_rng(42)

        # Generate starting points
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random points
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, self.dim))
        # Some structured points
        starts.append(np.ones(self.dim) * 2.0)
        starts.append(np.ones(self.dim) * -2.0)
        starts.append(np.array([1, 1, 1, 1, 1], dtype=float))
        starts.append(np.array([-1, -1, -1, -1, -1], dtype=float))

        # Shuffle starts to diversify
        idx = rng.permutation(len(starts))
        starts = [starts[i] for i in idx]

        for x0 in starts:
            if global_eval_count[0] >= budget_left:
                break
            if time.time() > deadline:
                break

            # Run L-BFGS-B from this start
            try:
                result = minimize(
                    counting_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': max(1, budget_left)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

            if global_eval_count[0] >= budget_left:
                break
            if time.time() > deadline:
                break

        # If we have budget left, do a final local refinement
        if best_x is not None and global_eval_count[0] < budget_left and time.time() < deadline:
            try:
                result = minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': max(1, budget_left)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
