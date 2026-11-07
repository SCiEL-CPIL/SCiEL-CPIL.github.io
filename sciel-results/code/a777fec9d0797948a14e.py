"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


class Optimizer:
    """Differential Evolution + L-BFGS-B polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            call_count += 1
            if call_count > self.budget:
                return best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with a small number of strategic points
        # Evaluate origin and a few random points to get initial estimates
        strategic_points = [
            np.zeros(self.dim),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
        ]
        for x in strategic_points:
            if call_count >= self.budget or time.time() > deadline - 30:
                break
            wrapped_func(x)

        # Phase 2: Differential Evolution for global search
        if call_count < self.budget - 50 and time.time() < deadline - 60:
            try:
                # Use a population size appropriate for the remaining budget
                # DE needs at least popsize * dim evaluations per generation
                # Let's use a moderate popsize
                popsize = 15
                # Max generations: leave room for polishing
                max_generations = 200
                
                # Run differential evolution
                # bounds is a list of tuples
                bounds = [(-5.0, 5.0)] * self.dim
                
                # We wrap the function to count calls and enforce budget
                # DE will call func many times. We need to stop if budget is exceeded.
                # The wrapped_func handles the count and returns best_f if over budget.
                # However, DE might not handle the "fake" return well if it expects convergence.
                # It will just keep trying. We rely on the time deadline and budget check inside wrapped_func.
                
                # To be safe, we can limit the number of function evaluations DE makes
                # by setting maxiter and tol, but the primary constraint is our call_count.
                
                res = differential_evolution(
                    wrapped_func,
                    bounds,
                    popsize=popsize,
                    maxiter=max_generations,
                    tol=1e-12,
                    atol=1e-14,
                    seed=42,
                    init='latinhypercube',
                    polish=False,  # We will do our own polish with L-BFGS-B
                    workers=1
                )
            except Exception:
                pass

        # Phase 3: Local refinement with L-BFGS-B from the best point found
        if best_x is not None and call_count < self.budget - 20 and time.time() < deadline - 30:
            try:
                res_lbfgs = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                pass

        # Phase 4: Final polish with Nelder-Mead if budget remains
        if best_x is not None and call_count < self.budget - 5 and time.time() < deadline - 10:
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14}
                )
            except Exception:
                pass

        # If we still have budget, try one more L-BFGS-B from a slightly perturbed best point
        if best_x is not None and call_count < self.budget - 5 and time.time() < deadline - 5:
            try:
                x_perturbed = best_x + rng.normal(0, 0.1, self.dim)
                x_perturbed = np.clip(x_perturbed, -5, 5)
                res_final = minimize(
                    wrapped_func,
                    x_perturbed,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                pass

        if best_x is not None:
            return best_x, best_f
        else:
            # Fallback: ensure at least one call was made
            if call_count == 0:
                x0 = np.zeros(self.dim)
                wrapped_func(x0)
            return np.zeros(self.dim), best_f

# EVOLVE-BLOCK-END
