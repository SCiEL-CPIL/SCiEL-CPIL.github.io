"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Nelder-Mead optimizer with strict budget control for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Strategy: Nelder-Mead from the center.
        # BBOB f4 is a shifted, rotated, and offset Büche-Rastrigin function.
        # The minimizer is in [-4, 4]^5. Starting at 0 is a good bet.
        # Nelder-Mead is efficient in low dimensions (dim=5).
        
        try:
            # Use Nelder-Mead with a large maxiter, but wrapped_func will stop it.
            # initial_simplex_size is important. Default is 0.00025 * |x| + 0.00025.
            # We can provide an initial simplex or let it default.
            # Let's try a standard call.
            res = minimize(
                wrapped_func,
                np.zeros(dim),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass

        # If we have budget left, try a few random restarts with Nelder-Mead
        # or just a local polish.
        # Given the 500 budget, Nelder-Mead might use a lot.
        # Let's see how many calls were used.
        
        if call_count < budget - 50:
            # Try a few random restarts
            rng = np.random.default_rng(42)
            for _ in range(5):
                if call_count >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                
                # Random point in [-4, 4]^5
                x0 = rng.uniform(-4, 4, size=dim)
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
