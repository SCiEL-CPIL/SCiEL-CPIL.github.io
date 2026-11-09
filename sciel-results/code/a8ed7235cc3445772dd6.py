"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Multi-start Nelder-Mead optimizer with Sobol exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise ValueError("Timeout")
            if calls >= total_budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                raise ValueError("Invalid shape")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite values")
                
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
                
            return f

        # Phase 1: Exploration using Sobol Sequence
        # 150 points is a good balance for 5D
        n_explore = 150
        try:
            sampler = qmc.Sobol(d=n, scramble=True)
            samples = sampler.random(n_explore)
            # Scale from [0, 1] to [-5, 5]
            samples = samples * 10.0 - 5.0
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(42)
            samples = rng.uniform(-5, 5, size=(n_explore, n))

        for x in samples:
            try:
                safe_func(x)
            except ValueError:
                break

        # Phase 2: Local Optimization using Nelder-Mead
        if calls < total_budget and not np.isinf(best_f):
            x_start = best_x.copy()
            
            # Construct an initial simplex centered at x_start
            # Ensure the simplex is not too small to escape local flatness
            simplex = [x_start.copy()]
            for i in range(n):
                x_point = x_start.copy()
                # Adaptive step size: larger if x_start is near 0
                step = 0.05 * abs(x_start[i]) + 0.5
                x_point[i] += step
                simplex.append(x_point)
            
            try:
                # Nelder-Mead is robust. 
                # maxiter 100 with ~5 evals/iter is ~500 evals.
                # We have ~350 calls left, so 70 maxiter is safe.
                res = minimize(safe_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': 70, 'xatol': 1e-4, 'fatol': 1e-4},
                               initial_simplex=simplex)
            except ValueError:
                # Budget or time exceeded, keep best found
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
