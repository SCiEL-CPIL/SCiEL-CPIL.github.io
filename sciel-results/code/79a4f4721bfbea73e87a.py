"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES style global search + L-BFGS-B local polish for Discus."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # We need to count function calls carefully
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def wrapped_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            call_count[0] += 1
            fx = func(x)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        # Strategy: Use a population-based derivative-free method for exploration,
        # then switch to L-BFGS-B for exploitation once we're close.

        # Phase 1: CMA-ES-like search (simplified)
        # Initialize population
        rng = np.random.default_rng(42)
        pop_size = 20
        x0 = rng.uniform(-4, 4, dim)
        sigma = 1.5

        # Simple (mu+lambda)-ES with adaptive sigma
        mu = max(1, pop_size // 2)
        pop = [x0 + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
        pop_f = [wrapped_func(np.clip(x, -5, 5)) for x in pop]

        # Sort by fitness
        sorted_idx = np.argsort(pop_f)
        pop = [pop[i] for i in sorted_idx]
        pop_f = [pop_f[i] for i in sorted_idx]

        # Track best
        x_best = pop[0].copy()
        f_best = pop_f[0]

        # Adaptive step size based on success rate
        sigma = 1.5
        for iteration in range(200):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            # Update center of mass
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * pop[i]

            # Update sigma based on success (simplified)
            # If best improved, increase sigma slightly; else decrease
            if f_best < best_f[0] + 1e-6:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)

            # Generate new population
            new_pop = []
            for _ in range(pop_size):
                if call_count[0] >= budget:
                    break
                y = m + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                new_pop.append(y)

            if not new_pop:
                break

            new_f = [wrapped_func(x) for x in new_pop]

            # Update best
            min_idx = np.argmin(new_f)
            if new_f[min_idx] < f_best:
                f_best = new_f[min_idx]
                x_best = new_pop[min_idx].copy()

            # Combine and sort
            combined_pop = pop + new_pop
            combined_f = pop_f + new_f
            combined_idx = np.argsort(combined_f)
            pop = [combined_pop[i] for i in combined_idx[:pop_size]]
            pop_f = [combined_f[i] for i in combined_idx[:pop_size]]

            # Check if we should switch to local optimization
            if f_best < 1.0 and call_count[0] < budget - 50:
                break

        # Phase 2: L-BFGS-B local polish
        if call_count[0] < budget - 10 and time.time() < deadline:
            x_start = x_best
            f_start = f_best

            # Use L-BFGS-B with bounds
            try:
                result = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < f_best:
                    f_best = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 3: Final fine-tuning with small steps
        if call_count[0] < budget and time.time() < deadline:
            sigma_final = 0.05
            for _ in range(budget - call_count[0]):
                if time.time() > deadline:
                    break
                y = x_best + sigma_final * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = wrapped_func(y)
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    sigma_final *= 0.9
                else:
                    sigma_final *= 0.8

        return x_best, f_best
# EVOLVE-BLOCK-END
