"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.elapsed_time = 0.0
        self.deadline = 290.0  # Safety margin before 330s kill

    def __call__(self, func):
        self.start_time = time.time()
        
        # 1. Global Exploration with Latin Hypercube Sampling
        # 50 samples to cover the space and identify promising regions near the ridge
        n_global = 50
        if n_global >= self.budget:
            # Fallback for very small budgets
            return self._random_search(func)

        sampler = qmc.LatinHypercube(d=self.dim, seed=0)
        points = sampler.random(n_global) * 10.0 - 5.0  # Scale to [-5, 5]
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = []
        
        # Evaluate global samples
        for x in points:
            if calls_made >= self.budget or self._check_time():
                break
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no points were evaluated
            x0 = np.zeros(self.dim)
            best_f = func(x0)
            calls_made += 1
            best_x = x0

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # 2. Local Refinement with Multi-start Nelder-Mead
        # Nelder-Mead is better than L-BFGS-B for flat ridges because it uses
        # geometric transformations that can slide along flat directions if the
        # simplex is not perfectly aligned, and 'adaptive' helps with scaling.
        
        # Select top K candidates for local search
        K = 5
        remaining_budget = self.budget - calls_made
        local_budget = remaining_budget // K if K > 0 else 0

        if local_budget < 10:
             # If budget is too tight, just return the best global point
             return self._best_x, self._best_f

        self._current_calls = calls_made
        self._max_calls = self.budget
        self._best_x = best_x
        self._best_f = best_f

        def wrapped_func(x):
            if self._check_time() or self._current_calls >= self._max_calls:
                return self._best_f
            self._current_calls += 1
            fx = func(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Run Nelder-Mead on top K candidates
        for i in range(min(K, len(candidates))):
            if self._current_calls >= self._max_calls or self._check_time():
                break
                
            start_x = candidates[i][1]
            
            try:
                res = minimize(
                    wrapped_func,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_budget,
                        'fatol': 1e-12,
                        'xatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return self._best_x, self._best_f

    def _random_search(self, func):
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(0)
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
