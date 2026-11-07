# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution + Nelder-Mead hybrid for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget

        # Strategy: Differential Evolution for global search, then Nelder-Mead for local polish
        # DE is robust to non-smoothness of the step ellipsoidal function

        # Phase 1: Differential Evolution (use ~70% of budget)
        de_budget = int(budget * 0.7)
        npop = 10  # Population size for DE
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5, 5, (npop, dim))
        fitness = np.array([count_eval(ind) for ind in population])

        # Track best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # DE loop
        for gen in range(100):
            if self._call_count >= de_budget or time.time() > deadline:
                break

            for i in range(npop):
                if self._call_count >= de_budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(
                    [j for j in range(npop) if j != i], size=3, replace=False
                )

                # Create trial vector using DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = count_eval(trial)

                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        # Phase 2: Nelder-Mead local search from best point (use remaining budget)
        remaining = budget - self._call_count
        if remaining > 10 and best_x is not None and time.time() < deadline:
            # Use scipy's Nelder-Mead
            from scipy.optimize import minimize

            def wrapper(x):
                if self._call_count >= budget:
                    return self._best_f
                x_clipped = np.clip(x, -5, 5)
                return count_eval(x_clipped)

            # Initial simplex size
            x0 = best_x.copy()
            options = {
                'maxiter': min(remaining // 2, 500),
                'maxfev': min(remaining, 500),
                'xatol': 1e-8,
                'fatol': 1e-10,
            }

            try:
                result = minimize(
                    wrapper, x0, method='Nelder-Mead',
                    options=options, timeout=max(1, deadline - time.time())
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Final polish with small perturbations
        if self._call_count < budget - 5 and best_x is not None and time.time() < deadline:
            for _ in range(min(20, budget - self._call_count - 2)):
                if self._call_count >= budget - 2 or time.time() > deadline:
                    break
                # Adaptive perturbation size based on current best
                scale = 0.05 * (1.0 + 0.1 * np.log1p(self._call_count))
                perturbation = rng.standard_normal(dim) * scale
                x_candidate = np.clip(best_x + perturbation, -5, 5)
                count_eval(x_candidate)

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
