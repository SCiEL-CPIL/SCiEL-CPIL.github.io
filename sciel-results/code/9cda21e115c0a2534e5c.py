"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple L-BFGS-B optimizer for BBOB f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf
        total_calls = 0

        def check_budget():
            nonlocal total_calls
            if total_calls >= budget or time.time() > deadline:
                return False
            return True

        # Use scipy's L-BFGS-B optimizer with multiple random restarts
        from scipy.optimize import minimize

        # Strategy: Use multiple random starting points and run L-BFGS-B from each
        # L-BFGS-B is efficient for smooth functions like bent cigar
        # We'll allocate budget across multiple restarts
        
        # First, evaluate center point
        center = np.zeros(dim)
        f_center = func(center)
        total_calls += 1
        best_f = f_center
        best_x = center.copy()

        # Number of restarts based on remaining budget
        # L-BFGS-B typically uses ~20-50 function evaluations per run
        # Let's try 10 restarts with ~40 evaluations each = 400 total
        n_restarts = 10
        evals_per_restart = (budget - 1) // n_restarts

        for r in range(n_restarts):
            if not check_budget():
                break
            
            # Random starting point
            x0 = rng.uniform(-4.5, 4.5, dim)
            
            # Count function evaluations for this run
            def objective(x):
                nonlocal total_calls, best_x, best_f
                if total_calls >= budget or time.time() > deadline:
                    return best_f  # Return current best to stop optimization
                total_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            # Run L-BFGS-B
            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': evals_per_restart}
                )
                # Update best from result
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
