"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: Sobol basin identification + BFGS refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Phase 1: Sobol sequence for initial exploration
        # We'll generate Sobol points manually using a simple LCG-based approach
        # or use scipy if available, otherwise fall back to stratified sampling
        
        # Try to use scipy's Sobol generator
        try:
            from scipy.stats import qmc
            sobol_gen = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sobol_gen.random(64)
            # Scale from [0,1]^d to [-5,5]^d
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            # Fallback: stratified random sampling
            rng = np.random.default_rng(42)
            sobol_points = rng.uniform(-5, 5, (64, self.dim))

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Evaluate Sobol points
        for x in sobol_points:
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: BFGS refinement from the best Sobol point
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                def wrapper(x):
                    nonlocal calls_made
                    if calls_made >= self.budget or time.time() > deadline:
                        raise ValueError("Budget exceeded")
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    return fx

                # Run BFGS from the best point found
                result = minimize(
                    wrapper,
                    best_x,
                    method='BFGS',
                    options={'maxiter': self.budget - calls_made,
                             'gtol': 1e-6,
                             'disp': False}
                )

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                # If BFGS fails, continue with the best we have
                pass

        # Phase 3: If we still have budget, do a local search with adaptive step
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            rng = np.random.default_rng(123)
            sigma = 0.5
            for _ in range(self.budget - calls_made):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                y = best_x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
