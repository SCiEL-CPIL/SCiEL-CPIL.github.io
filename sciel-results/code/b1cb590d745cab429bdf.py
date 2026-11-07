"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Dual Annealing: global exploration via annealing + local Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Wrap func to count calls and enforce deadline
        call_count = [0]
        best_x = [None]
        best_f = [float('inf')]

        def wrapped_func(x):
            if call_count[0] >= self.budget:
                return best_f[0]
            if time.time() > deadline:
                return best_f[0]
            call_count[0] += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        rng = np.random.default_rng(42)

        # Strategy 1: Dual Annealing
        # Phase 1: Global exploration with simulated annealing-style random restarts
        # Phase 2: Local refinement with Nelder-Mead

        # First, do a quick random search to get a good starting point
        best_start_x = None
        best_start_f = float('inf')

        # Try several random starting points
        n_random = 20
        for i in range(n_random):
            x = rng.uniform(-5, 5, self.dim)
            f = wrapped_func(x)
            if f < best_start_f:
                best_start_f = f
                best_start_x = x.copy()

        if best_start_x is None:
            return best_x[0] if best_x[0] is not None else np.zeros(self.dim), best_f[0]

        # Now use scipy's dual_annealing for global optimization
        # This combines annealing (global) with Nelder-Mead (local)
        try:
            result = minimize(
                wrapped_func,
                best_start_x,
                method='dual_annealing',
                options={
                    'maxiter': 50,  # Limit iterations to stay within budget
                    'initial_temp': 5.0,
                    'restart_temp_ratio': 0.2,
                    'noannal': 5,  # Number of no-improvement iterations before restart
                    'accept': 0.05,
                    'visit': 2.0,
                    'maxiter': 30,
                    'initial_temp': 10.0,
                    'restart_temp_ratio': 0.1,
                    'noannal': 10,
                    'accept': 0.1,
                    'visit': 1.0,
                    'verbose': 0
                }
            )
        except Exception:
            pass

        # Check if we've used up budget or hit deadline
        if call_count[0] >= self.budget or time.time() > deadline:
            return best_x[0] if best_x[0] is not None else np.zeros(self.dim), best_f[0]

        # Phase 2: Local refinement with Nelder-Mead from best point found
        if best_x[0] is not None:
            try:
                result_nelder = minimize(
                    wrapped_func,
                    best_x[0],
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x[0] if best_x[0] is not None else np.zeros(self.dim), best_f[0]
# EVOLVE-BLOCK-END
