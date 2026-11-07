"""Rewritten optimizer using CMA-ES with adaptive population and restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Run CMA-ES with given initial point and sigma."""
        dim = self.dim
        N = max(4, int(4 + 3 * np.log(dim)))  # population size
        # CMA-ES parameters
        c_sigma = (N + 2) / (dim + N + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((N - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + N / dim) / (dim + 4 + 2 * N / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + N)
        c_mu = min(1 - c_1, 2 * ((N / 2 + 1) / (dim + 2) - 1))

        x = x0.copy()
        sigma = sigma0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Precompute weights
        mu = N // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        while calls < budget:
            # Generate population
            B = rng.standard_normal((N, dim))
            # Apply evolution path and covariance
            # For simplicity, use standard normal and scale by sigma
            # Full CMA-ES would update C, but for speed we use simplified version
            offspring = np.array([x + sigma * b for b in B])
            offspring = np.clip(offspring, -5, 5)

            # Evaluate
            fitnesses = np.empty(N)
            for i in range(N):
                fitnesses[i] = func(offspring[i])
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                # Check if any improved best
                for i in range(N):
                    if fitnesses[i] < best_f:
                        best_f = fitnesses[i]
                        best_x = offspring[i].copy()
                break

            # Sort by fitness
            order = np.argsort(fitnesses)
            y_sorted = offspring[order]
            f_sorted = fitnesses[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Compute weighted mean
            y_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update sigma (simplified: use success rate)
            # Count successes
            success = np.sum(f_sorted[:mu] < f_sorted[mu]) if mu < N else 0
            sigma *= np.exp(c_sigma / d_sigma * (success / mu - 0.5))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update mean
            x = np.clip(y_mean, -5, 5)

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start with different initial points and sigmas
        # Use a few restarts to escape local optima (important for f22)
        n_starts = 3
        calls_per_start = budget // n_starts
        extra = budget % n_starts

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Initial points: center, random, and a few structured points
        initial_points = [
            np.zeros(dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-3, 3, dim),
        ]

        initial_sigmas = [2.0, 1.5, 1.0]

        for i in range(n_starts):
            x0 = initial_points[i].copy()
            sigma0 = initial_sigmas[i]
            alloc = calls_per_start + (1 if i < extra else 0)

            if alloc <= 1:
                break

            x_best, f_best, calls_used = self._cma_es(
                func, x0, sigma0, alloc, rng
            )
            total_calls += calls_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

            if total_calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
