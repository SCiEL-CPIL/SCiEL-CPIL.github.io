import numpy as np
import scipy.optimize
import time

class Optimizer:
    """Optimizer using Nelder-Mead with random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 295.0  # 5s safety margin

        dim = self.dim
        budget = self.budget

        # Safe wrapper for func to handle exceptions and non-finite values
        def safe_func(x):
            try:
                # Clip to the box [-5, 5]^dim
                x_clipped = np.clip(x, -5, 5)
                val = float(func(x_clipped))
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Number of restarts
        # With budget 300 and dim 5, we can afford 4 restarts of ~75 evals each.
        n_restarts = 4
        budget_per_restart = budget // n_restarts

        rng = np.random.default_rng(42)

        for i in range(n_restarts):
            if time.time() - start_time > timeout:
                break

            # Choose initial point
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            # Create initial simplex
            # We want a simplex that covers a reasonable area.
            # A size of 1.0 in each dimension is a good start for [-5, 5].
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for j in range(1, dim + 1):
                simplex[j] = x0.copy()
                simplex[j][j - 1] += 1.0

            # Clip simplex to box
            simplex = np.clip(simplex, -5, 5)

            # Determine max iterations
            # Nelder-Mead uses dim+1 evaluations for the initial simplex.
            # Each iteration uses 1 evaluation.
            max_iter = max(1, budget_per_restart - (dim + 1))

            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'initial_simplex': simplex
                    }
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x

            except Exception:
                pass

        return best_x, best_f
