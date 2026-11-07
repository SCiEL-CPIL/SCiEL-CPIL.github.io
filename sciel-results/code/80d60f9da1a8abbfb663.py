"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Hybrid optimizer: CMA-ES-like with adaptive restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        # State tracking
        best_x = None
        best_f = np.inf
        calls_used = 0

        def call_func(x):
            nonlocal calls_used
            if time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            calls_used += 1
            return fx

        # Phase 1: Initial sampling to get a good starting point
        # Use a stratified/latin-hypercube-like approach for initial samples
        rng = np.random.default_rng(42)

        # First, try center and a few diverse points
        initial_points = [np.zeros(self.dim)]
        for _ in range(10):
            initial_points.append(rng.uniform(-5, 5, self.dim))

        for x0 in initial_points:
            if calls_used >= self.budget or time.time() > deadline:
                break
            fx = call_func(x0)
            if fx is None:
                continue
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: CMA-ES-like search with adaptive step size
        # We'll use a simplified CMA-ES with population-based updates
        # But to keep it simple and fast, we'll use a (lambda, mu)-ES with adaptive sigma

        # Parameters
        pop_size = max(4, self.dim + 3)
        sigma = 1.5
        success_ratio = 1.0 / 5.0
        sigma_inc = 1.5
        sigma_dec = 0.87

        # Adaptive restart: if no improvement for N steps, restart with smaller sigma
        no_improve_count = 0
        max_no_improve = 50

        while calls_used < self.budget and time.time() < deadline:
            # Generate population
            population = []
            for _ in range(pop_size):
                x_new = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                population.append(x_new)

            # Evaluate population
            fitnesses = []
            for x_new in population:
                if calls_used >= self.budget or time.time() > deadline:
                    break
                fx = call_func(x_new)
                if fx is None:
                    break
                fitnesses.append((fx, x_new))

            if not fitnesses:
                break

            # Sort by fitness
            fitnesses.sort(key=lambda t: t[0])
            best_fx, best_x_new = fitnesses[0]

            # Update best
            improved = False
            if best_fx < best_f:
                best_f = best_fx
                best_x = best_x_new.copy()
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive sigma update
            # Count how many in the population were better than the previous best
            better_count = sum(1 for fx, _ in fitnesses if fx < best_f + 1e-10)
            if better_count >= pop_size * success_ratio:
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec

            # Adaptive restart: if stuck, try a local restart
            if no_improve_count > max_no_improve:
                # Try a few random restarts around the current best
                for _ in range(3):
                    if calls_used >= self.budget or time.time() > deadline:
                        break
                    x_restart = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                    fx_restart = call_func(x_restart)
                    if fx_restart is not None and fx_restart < best_f:
                        best_f = fx_restart
                        best_x = x_restart.copy()
                        no_improve_count = 0
                        sigma = 0.5
                        break
                no_improve_count = 0
                sigma = max(0.1, sigma * 0.5)

            # If sigma is too small, we're likely converged
            if sigma < 1e-6:
                break

        # Phase 3: Local polish with Nelder-Mead or BFGS
        if best_x is not None and calls_used < self.budget - 10 and time.time() < deadline:
            try:
                # Use Nelder-Mead for local optimization
                result = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(100, self.budget - calls_used),
                             'xatol': 1e-8, 'fatol': 1e-10}
                )
                # Update best if improved
                fx_result = float(result.fun)
                if fx_result < best_f:
                    best_f = fx_result
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
