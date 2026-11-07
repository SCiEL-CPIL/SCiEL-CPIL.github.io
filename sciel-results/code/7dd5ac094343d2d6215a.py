# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallelized multi-start differential evolution (DE/rand/1) with
    elite preservation, adaptive sigma and a final local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _batch_eval(self, func, X):
        """Evaluate a batch of points; returns array of objective values."""
        n = X.shape[0]
        vals = np.empty(n)
        for i in range(n):
            vals[i] = func(X[i])
        return vals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0

        def time_ok():
            return time.time() < deadline

        # ---- Phase 0: stratified random initialization (cheap, diverse) ----
        # 50 candidates spread across the box so the population starts global.
        n_pop = 50
        grid = np.linspace(-5.0, 5.0, 8)
        X = np.empty((n_pop, dim))
        for i in range(n_pop):
            X[i] = grid[rng.integers(0, len(grid), size=dim)]
        F = self._batch_eval(func, X)
        calls += n_pop

        best_idx = int(np.argmin(F))
        best_x = X[best_idx].copy()
        best_f = float(F[best_idx])

        # ---- Phase 1: DE/rand/1 main loop ----
        sigma = 1.5
        F_factor = 0.8
        CR = 0.5
        n_elite = 4
        min_sigma, max_sigma = 0.02, 4.0

        while calls < budget - 40 and time_ok():
            # build mutation base: random parent + diff of two random parents
            base = rng.integers(0, n_pop, size=(n_pop, dim))
            a = rng.integers(0, n_pop, size=(n_pop, dim))
            b = rng.integers(0, n_pop, size=(n_pop, dim))
            v = X[base] + sigma * F_factor * (X[a] - X[b])

            # binomial crossover (per-dimension) with random crossover mask
            mask = (rng.random((n_pop, dim)) < CR).astype(float)
            # guarantee at least one dimension taken from v
            jrand = rng.integers(0, dim, size=n_pop)
            mask[np.arange(n_pop), jrand] = 1.0
            trial = np.where(mask, v, X)
            trial = np.clip(trial, -5.0, 5.0)

            F_trial = self._batch_eval(func, trial)
            calls += n_pop

            # selection: keep the better of parent vs trial per individual
            keep = F_trial < F
            X[keep] = trial[keep]
            F[keep] = F_trial[keep]
            n_imp = int(keep.sum())

            # update global best
            gbest_idx = int(np.argmin(F))
            if F[gbest_idx] < best_f:
                best_f = float(F[gbest_idx])
                best_x = X[gbest_idx].copy()

            # adaptive sigma: target ~10% success per generation
            if n_imp > 0.15 * n_pop:
                sigma *= 1.05
            elif n_imp < 0.05 * n_pop:
                sigma *= 0.85
            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            # elite preservation: re-seed worst slots from the best to keep
            # diversity while guaranteeing the top solutions persist
            order = np.argsort(F)
            elites = order[:n_elite]
            worst = order[-n_elite:]
            for k in range(n_elite):
                X[worst[k]] = X[elites[k]]
                F[worst[k]] = F[elites[k]]

        # ---- Phase 2: local polish (1+1-ES) around the best point ----
        remaining = budget - calls
        if remaining > 5 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            for _ in range(remaining):
                if not time_ok():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.88
                sigma = float(np.clip(sigma, 0.003, 2.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
