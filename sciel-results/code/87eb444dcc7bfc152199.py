"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with adaptive parameters and multi-start."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _differential_evolution(self, func, x0, budget, rng):
        """Run DE with adaptive F and CR. Returns (best_x, best_f, calls_made)."""
        n = self.dim
        pop_size = max(8, 4 * n)
        pop_size = min(pop_size, max(4, budget // 2))

        # Initialize population around x0
        pop = np.empty((pop_size, n))
        pop[0] = x0
        for i in range(1, pop_size):
            spread = 2.0 * (1 - i / pop_size) + 0.5
            pop[i] = x0 + spread * rng.standard_normal(n)
            pop[i] = np.clip(pop[i], -5, 5)

        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        calls = pop_size

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Adaptive parameters
        F = 0.8
        CR = 0.5
        success_streak = 0
        no_improve_count = 0

        while calls < budget:
            # Adaptive F and CR
            if success_streak > 3:
                F = max(0.4, F * 0.95)
                CR = min(0.9, CR * 1.05)
            elif no_improve_count > 10:
                F = min(1.0, F * 1.1)
                CR = max(0.3, CR * 0.95)

            # Generate offspring
            for i in range(pop_size):
                if calls >= budget:
                    break

                # Select three distinct random indices, none equal to i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Differential mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, -5, 5)

                # Binomial crossover
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == rng.integers(n):
                        trial[j] = mutant[j]

                # Evaluate
                f_trial = func(trial)
                calls += 1

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        success_streak += 1
                        no_improve_count = 0
                    else:
                        success_streak = max(0, success_streak - 1)
                else:
                    no_improve_count += 1

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Multi-start with diverse initializations
        num_restarts = 4
        budget_per_restart = budget // num_restarts

        starts = [
            np.zeros(n),
            rng.uniform(-3, 3, n),
            rng.uniform(-4, 4, n),
            rng.uniform(-2, 2, n),
        ]

        for restart in range(num_restarts):
            if calls_used >= budget:
                break

            x0 = np.clip(starts[restart], -5, 5)
            remaining = budget - calls_used
            alloc = min(budget_per_restart, remaining)

            # Run DE (it handles its own initial evaluation)
            x_opt, f_opt, calls_in_run = self._differential_evolution(
                func, x0, alloc, rng
            )
            calls_used += calls_in_run

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
