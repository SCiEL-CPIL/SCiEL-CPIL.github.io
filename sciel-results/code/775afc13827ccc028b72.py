"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time

class Optimizer:
    """Hybrid Sobol sampling and L-BFGS-B optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.elapsed_time = 0.0
        self.deadline = 300.0  # Safety margin before 330s kill

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(0)
        
        # 1. Global Exploration with Sobol Sequence
        # We use a fixed number of samples for global coverage.
        # 100 samples is a good balance for dim=5.
        n_global = 100
        if n_global >= self.budget:
            # If budget is very small, just do random search
            return self._random_search(func, rng)

        sampler = qmc.LatinHypercube(d=self.dim, seed=0)
        points = sampler.random(n_global) * 10.0 - 5.0  # Scale to [-5, 5]
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate global samples
        for x in points:
            if calls_made >= self.budget or self._check_time():
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no points were evaluated
            x0 = rng.uniform(-5, 5, self.dim)
            best_x = x0
            best_f = func(x0)
            calls_made += 1

        # 2. Local Refinement with L-BFGS-B
        # We need to wrap the function to track calls and enforce budget/deadline
        self._current_calls = calls_made
        self._max_calls = self.budget
        self._best_x = best_x
        self._best_f = best_f

        def wrapped_func(x):
            if self._check_time() or self._current_calls >= self._max_calls:
                return self._best_f  # Return best known to stop optimization
            self._current_calls += 1
            fx = func(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Define bounds
        bounds = [(-5.0, 5.0)] * self.dim

        # Run L-BFGS-B
        try:
            res = minimize(
                wrapped_func,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            pass

        # Check if we have remaining budget for other local starts
        # If L-BFGS-B didn't use all budget, try a few other top candidates from Sobol
        # But for simplicity and speed, we stick to the best one. 
        # The sharp ridge is usually found by the best Sobol point if n_global is sufficient.
        
        return self._best_x, self._best_f

    def _random_search(self, func, rng):
        best_x = None
        best_f = np.inf
        for _ in range(self.budget):
            if self._check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = 0.0
        return best_x, best_f

    def _check_time(self):
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        self.elapsed_time = elapsed
        return elapsed > self.deadline
# EVOLVE-BLOCK-END
