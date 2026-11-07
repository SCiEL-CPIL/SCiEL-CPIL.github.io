"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Portfolio optimizer: multi-start (1+1)-ES + L-BFGS-B refinement for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        dim = self.dim
        rng = np.random.default_rng(0)

        calls = [0]
        best_x = None
        best_fx = float('inf')

        def call_func(x):
            nonlocal best_x, best_fx
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(dim)
            calls[0] += 1
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start (1+1)-ES with adaptive step size
        n_restarts = 8
        budget_per_restart = max(10, (self.budget - dim * n_restarts) // n_restarts)
        
        for r in range(n_restarts):
            if calls[0] >= self.budget or time.time() > deadline - 60:
                break

            # Random start
            x = rng.uniform(-4.5, 4.5, dim)
            fx = call_func(x)

            # (1+1)-ES with 1/5 success rule
            sigma = 1.0
            success_count = 0
            total_steps = 0
            max_steps = budget_per_restart

            for step in range(max_steps):
                if calls[0] >= self.budget or time.time() > deadline - 60:
                    break

                # Generate trial point
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                fx_trial = call_func(x_trial)

                total_steps += 1
                if fx_trial <= fx:
                    x = x_trial
                    fx = fx_trial
                    success_count += 1
                    # Increase sigma on success
                    sigma *= 1.5
                else:
                    # Decrease sigma on failure
                    sigma *= 1.0 / 1.5

                # 1/5 rule: adjust sigma every 1/dim steps
                if total_steps % dim == 0:
                    rate = success_count / total_steps
                    if rate > 1.0 / 5:
                        sigma *= 1.2
                    elif rate < 1.0 / 5:
                        sigma *= 1.0 / 1.2

                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Early stop if converged
                if sigma < 1e-6:
                    break

        # Phase 2: L-BFGS-B refinement from best point found
        if best_x is not None and calls[0] < self.budget and time.time() < deadline - 30:
            def wrapper(x):
                if calls[0] >= self.budget or time.time() > deadline - 10:
                    return best_fx  # Return current best to stop optimization
                return call_func(x)

            # Use numerical gradient approximation
            try:
                res = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': max(10, self.budget - calls[0] - 5),
                             'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final local polish with very small steps
        if best_x is not None and calls[0] < self.budget and time.time() < deadline - 5:
            sigma = 0.01
            for _ in range(min(50, self.budget - calls[0])):
                if calls[0] >= self.budget or time.time() > deadline - 2:
                    break
                z = rng.standard_normal(dim)
                x_trial = best_x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                fx_trial = call_func(x_trial)

        return best_x, best_fx
# EVOLVE-BLOCK-END
