"""Rewritten optimizer for BBOB f6 (Attractive sector) using Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Continuous optimizer using Differential Evolution."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        x_best = None
        f_best = np.inf

        def eval_func(x):
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        # Population size
        N = 15
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        f_pop = np.array([eval_func(x) for x in pop])

        # Track best
        if x_best is None:
            idx = np.argmin(f_pop)
            x_best = pop[idx].copy()
            f_best = f_pop[idx]

        # Differential Evolution Loop
        while calls_made < budget:
            # Generate trial vectors for each individual
            for i in range(N):
                if calls_made >= budget:
                    break
                
                # Select three distinct random indices different from i
                candidates = [j for j in range(N) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                
                x_a = pop[a]
                x_b = pop[b]
                x_c = pop[c]
                
                # Mutation
                v = x_a + F * (x_b - x_c)
                
                # Crossover (Binomial)
                r = rng.random(dim)
                j_rand = rng.integers(0, dim)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                
                # Bound check
                trial = np.clip(trial, -5, 5)
                
                # Evaluation
                f_trial = eval_func(trial)
                
                # Selection
                if f_trial <= f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial

        return x_best, f_best
# EVOLVE-BLOCK-END
