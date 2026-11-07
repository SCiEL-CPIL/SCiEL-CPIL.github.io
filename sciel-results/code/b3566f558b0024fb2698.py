"""Optimizer for BBOB-style Rosenbrock (f8) with hidden transformations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES-like global search followed by L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Wrap func to count calls and track best
        state = {'calls': 0, 'best_x': None, 'best_f': np.inf}

        def wrapped_func(x):
            state['calls'] += 1
            if state['calls'] > self.budget:
                raise RuntimeError("Budget exceeded")
            fx = func(x)
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Global search with a population-based method
        # Use a simple CMA-ES-like approach with restarts
        best_overall_x = None
        best_overall_f = np.inf

        # Try multiple starting points with different strategies
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts

        for restart in range(num_restarts):
            if state['calls'] >= self.budget:
                break
            if time.time() > deadline - 30:
                break

            # Different initialization strategies
            if restart == 0:
                x0 = np.zeros(self.dim)
            elif restart == 1:
                x0 = rng.uniform(-4, 4, self.dim)
            elif restart == 2:
                x0 = rng.uniform(-5, 5, self.dim)
            elif restart == 3:
                x0 = rng.normal(0, 2, self.dim)
            else:
                x0 = rng.uniform(-3, 3, self.dim)

            # Run a local search from this point using L-BFGS-B
            # This is fast and effective for smooth functions like Rosenbrock
            try:
                result = scipy_minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': budget_per_restart}
                )
                if result.fun < best_overall_f:
                    best_overall_f = result.fun
                    best_overall_x = result.x.copy()
            except Exception:
                pass

            # Check time
            if time.time() > deadline - 60:
                break

        # Phase 2: Polish with more aggressive local optimization
        if best_overall_x is not None and state['calls'] < self.budget:
            remaining_budget = self.budget - state['calls']
            if remaining_budget > 10 and time.time() < deadline - 30:
                try:
                    result = scipy_minimize(
                        wrapped_func,
                        best_overall_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 500, 'maxfun': remaining_budget,
                                 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                    if result.fun < best_overall_f:
                        best_overall_f = result.fun
                        best_overall_x = result.x.copy()
                except Exception:
                    pass

        # Phase 3: Final fine-tuning with Nelder-Mead if budget remains
        if best_overall_x is not None and state['calls'] < self.budget:
            remaining_budget = self.budget - state['calls']
            if remaining_budget > 5 and time.time() < deadline - 10:
                try:
                    result = scipy_minimize(
                        wrapped_func,
                        best_overall_x,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'maxfev': remaining_budget,
                                 'xatol': 1e-12, 'fatol': 1e-12}
                    )
                    if result.fun < best_overall_f:
                        best_overall_f = result.fun
                        best_overall_x = result.x.copy()
                except Exception:
                    pass

        # Return the best point found
        if best_overall_x is None:
            best_overall_x = np.zeros(self.dim)

        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
