"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (1+lambda) ES with adaptive step size for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic Halton initialization for early progress
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim):
            bases = [2, 3, 5, 7, 11]
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = -5.0 + 10.0 * halton(index + 1, bases[d])
            return x

        n_halton = min(30, max(10, budget // 15))
        for i in range(n_halton):
            if time.time() > deadline:
                break
            if calls_made >= budget - 50:
                break
            x = halton_point(i, dim)
            x = np.clip(x, -5, 5)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return None, float('inf')

        # Phase 2: (1+lambda) Evolutionary Strategy with adaptive sigma
        rng = np.random.default_rng(42)
        x_best = best_x.copy()
        f_best = best_f
        sigma = 0.8  # initial step size
        lambda_pop = 8  # population size
        sigma_increase = 1.5
        sigma_decrease = 0.7
        sigma_min = 1e-6
        sigma_max = 3.0
        stagnation_count = 0
        max_stagnation = 20

        while calls_made < budget - 5 and time.time() < deadline:
            # Generate offspring
            offspring = []
            for j in range(lambda_pop):
                if calls_made >= budget - 5:
                    break
                y = x_best + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if fy is None:
                    break
                offspring.append((y, fy))

            if not offspring:
                break

            # Find best offspring
            best_offspring = min(offspring, key=lambda t: t[1])
            y_best, fy_best = best_offspring

            if fy_best < f_best - 1e-10:
                # Improvement
                x_best = y_best.copy()
                f_best = fy_best
                best_x = x_best.copy()
                best_f = f_best
                sigma = min(sigma * sigma_increase, sigma_max)
                stagnation_count = 0
            else:
                # No improvement
                stagnation_count += 1
                sigma = max(sigma * sigma_decrease, sigma_min)
                
                # If stuck for too long, try a larger exploration step
                if stagnation_count > max_stagnation:
                    sigma = min(sigma * 3.0, sigma_max)
                    stagnation_count = 0

            # Occasionally try a restart from a different Halton point if stuck
            if stagnation_count > 5 and calls_made < budget - 30:
                restart_idx = n_halton + (stagnation_count // 3) * 7
                x_restart = halton_point(restart_idx, dim)
                x_restart = np.clip(x_restart, -5, 5)
                fr = eval_func(x_restart)
                if fr is None:
                    break
                if fr < f_best - 1e-10:
                    x_best = x_restart.copy()
                    f_best = fr
                    best_x = x_best.copy()
                    best_f = f_best
                    sigma = 1.0
                    stagnation_count = 0

        # Phase 3: Final local refinement with smaller steps
        if best_x is not None and calls_made < budget - 2:
            sigma_final = 0.05
            for _ in range(10):
                if time.time() > deadline or calls_made >= budget - 2:
                    break
                y = best_x + sigma_final * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy

        return best_x, best_f
# EVOLVE-BLOCK-END
