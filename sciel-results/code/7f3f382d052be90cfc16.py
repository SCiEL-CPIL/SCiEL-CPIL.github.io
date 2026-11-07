"""Fresh design: population-based adaptive search with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive population-based optimizer for the sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(12345)
        dim = self.dim

        # Initial broad sample to quickly locate the basin of attraction.
        n_init = min(40, self.budget)
        X = rng.uniform(-5.0, 5.0, size=(n_init, dim))
        F = np.empty(n_init)
        for i in range(n_init):
            F[i] = func(X[i])

        idx = int(np.argmin(F))
        x = X[idx].copy()
        fx = F[idx]

        # Estimate initial step size from the spread of good points.
        order = np.argsort(F)
        n_top = max(3, n_init // 2)
        spread = X[order[:n_top]]
        sigma = max(0.05, float(np.std(spread, axis=0).mean()))

        # Adaptive population size and step size.
        pop = max(4, dim + 2)
        success = 0.0

        calls = n_init
        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate population around current best.
            X = x + sigma * rng.standard_normal((pop, dim))
            X = np.clip(X, -5.0, 5.5)
            F = np.empty(pop)
            for i in range(pop):
                F[i] = func(X[i])
                calls += 1
                if calls >= self.budget:
                    break
                if (i % 8 == 0) and time.time() > deadline:
                    break

            # Evaluate only what was generated.
            n_eval = min(pop, calls - n_init)
            if n_eval <= 0:
                break

            best_i = int(np.argmin(F[:n_eval]))
            best_f = F[best_i]
            best_x = X[best_i].copy()

            if best_f < fx:
                success = 0.6 * success + 0.4
                x = best_x
                fx = best_f
            else:
                success = 0.6 * success + 0.0

            # Adapt step size based on success rate.
            if success > 0.3:
                sigma *= 1.25
            elif success < 0.1:
                sigma *= 0.7
            else:
                sigma *= 0.98

            sigma = min(max(sigma, 1e-4), 4.0)

            # Occasionally add a global exploratory point to escape plateaus.
            if (calls % 20) == 0 and calls < self.budget - 1:
                gx = rng.uniform(-5.0, 5.0, size=dim)
                gf = func(gx)
                calls += 1
                if gf < fx:
                    x = gx
                    fx = gf

        return x, fx
# EVOLVE-BLOCK-END
