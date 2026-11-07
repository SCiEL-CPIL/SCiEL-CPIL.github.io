"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (mu+lambda)-ES with stagnation-triggered restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Parameters for (mu+lambda)-ES
        mu = 8
        lam = 32
        sigma = 2.0
        stagnation_limit = 80  # generations without improvement before restart

        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start exploration with (mu+lambda)-ES
        # We'll do multiple independent runs, restarting when stagnated
        num_restarts = 5
        per_restart_budget = max(1, (budget - 50) // num_restarts)

        for restart in range(num_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize population
            if restart == 0:
                # First restart: diverse initial points
                x_center = rng.uniform(-4, 4, dim)
            else:
                # Subsequent restarts: perturb around best found so far
                if best_x is not None:
                    x_center = best_x + rng.normal(0, 1.0, dim)
                    x_center = np.clip(x_center, -5, 5)
                else:
                    x_center = rng.uniform(-5, 5, dim)

            # Population: mu parents, lambda offspring
            # Start with a population around x_center
            parents = np.array([x_center + rng.normal(0, sigma * 0.5, dim) for _ in range(mu)])
            parents = np.clip(parents, -5, 5)

            # Evaluate initial parents
            parent_f = np.array([evaluate(p) for p in parents])
            if calls_used >= budget or time.time() > deadline:
                break

            sigma = 2.0
            gen = 0
            best_gen_f = best_f
            no_improve = 0

            while calls_used < budget and time.time() < deadline:
                gen += 1
                # Generate lambda offspring
                offspring = np.array([
                    parents[i % mu] + sigma * rng.standard_normal(dim)
                    for i in range(lam)
                ])
                offspring = np.clip(offspring, -5, 5)

                # Evaluate offspring
                off_f = np.array([evaluate(o) for o in offspring])
                if calls_used >= budget or time.time() > deadline:
                    break

                # Combine parents and offspring, select best mu
                combined = np.vstack([parents, offspring])
                combined_f = np.concatenate([parent_f, off_f])
                idx = np.argsort(combined_f)[:mu]
                parents = combined[idx]
                parent_f = combined_f[idx]

                # Check for improvement
                if parent_f[0] < best_gen_f - 1e-10:
                    best_gen_f = parent_f[0]
                    no_improve = 0
                    # Adapt sigma: increase if improving
                    sigma *= 1.1
                else:
                    no_improve += 1
                    sigma *= 0.9

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

                # Stagnation check
                if no_improve >= stagnation_limit:
                    break

                if calls_used >= budget or time.time() > deadline:
                    break

        # Phase 2: Local refinement with (1+1)-ES around best found
        if calls_used < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - calls_used):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                sigma = max(sigma, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
