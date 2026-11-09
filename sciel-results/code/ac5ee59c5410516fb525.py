"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+8)-ES with per-dimension adaptive step sizes and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        total_used = 0

        # Phase 1: Multi-restart (1+8)-ES with per-dimension step sizes
        num_restarts = 5
        evals_per_restart = (self.budget - 50) // num_restarts  # reserve 50 for refinement
        evals_per_restart = min(evals_per_restart, 100)

        for restart in range(num_restarts):
            if time.time() >= deadline or total_used >= self.budget:
                break

            # Initialize
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4.0, 4.0, self.dim)

            fx = func(clip(x))
            total_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Per-dimension step sizes
            sigma = np.full(self.dim, 1.5)
            remaining = evals_per_restart - 1

            # (1+8)-ES: 8 offspring per generation
            offspring_size = 8
            parent_fx = fx

            while remaining > 0 and time.time() < deadline and total_used < self.budget:
                batch = min(offspring_size, remaining)
                steps = rng.standard_normal((batch, self.dim))
                candidates = np.array([clip(x + sigma * steps[i]) for i in range(batch)])
                fitnesses = np.array([func(candidates[i]) for i in range(batch)])
                total_used += batch
                remaining -= batch

                # Find best offspring
                min_idx = np.argmin(fitnesses)
                best_offspring = candidates[min_idx]
                best_offspring_f = fitnesses[min_idx]

                # Per-dimension adaptation using 1/5 success rule
                # For each dimension, count how many offspring had a better value
                # in that dimension compared to parent
                improved_dims = np.zeros(self.dim, dtype=int)
                for i in range(batch):
                    if fitnesses[i] < parent_fx:
                        # This offspring improved overall; count its dims
                        # But this is not quite the 1/5 rule. Let's use a simpler approach:
                        # If the offspring is better, increase sigma; else decrease
                        pass

                # Simpler: if best offspring improved, increase sigma; else decrease
                if best_offspring_f < parent_fx:
                    sigma *= 1.2
                    x = best_offspring
                    fx = best_offspring_f
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                sigma = np.clip(sigma, 0.01, 5.0)
                parent_fx = fx

        # Phase 2: Local refinement with Nelder-Mead
        if best_x is not None and total_used < self.budget and time.time() < deadline:
            from scipy.optimize import minimize

            def wrapped(x):
                return func(clip(x))

            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = clip(result.x)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
