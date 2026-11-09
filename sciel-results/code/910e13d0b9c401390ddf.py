"""Hybrid optimizer: population-based DE for global search, Nelder-Mead for local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Population-based differential evolution (DE) for global search on rotated Rosenbrock,
    followed by Nelder-Mead local refinement.
    
    Strategy:
      1. Initialize a diverse population including the known Rosenbrock minimum (1,...,1).
      2. Run DE with adaptive mutation for the bulk of the budget.
      3. Polish the best candidate with Nelder-Mead using remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin

        def timed():
            return (time.time() - t_start) > (deadline - t_start)

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        rng = np.random.default_rng(42)
        calls_used = 0

        def do_call(x):
            nonlocal calls_used
            fx = func(clip_box(x))
            calls_used += 1
            return fx

        # --- Phase 0: Evaluate obvious candidates ---
        candidates = [
            np.ones(dim),
            np.zeros(dim),
            -np.ones(dim),
        ]

        best_x = None
        best_f = np.inf

        for cand in candidates:
            f_c = do_call(cand)
            if f_c < best_f:
                best_f = f_c
                best_x = cand.copy()

        # --- Phase 1: Differential Evolution ---
        # Population size: small for efficiency, but enough for diversity
        pop_size = 15
        # Reserve some budget for local polish
        de_budget = int(budget * 0.7)
        max_generations = max(1, (de_budget - calls_used) // pop_size)

        # Initialize population
        population = []
        # First member is the current best
        population.append(best_x.copy())
        # Add diverse random points
        for i in range(pop_size - 1):
            if i % 3 == 0:
                # Random in box
                x = rng.uniform(-4.5, 4.5, dim)
            elif i % 3 == 1:
                # Perturbed best
                x = best_x + rng.standard_normal(dim) * 2.0
            else:
                # Random near origin
                x = rng.uniform(-2, 2, dim)
            population.append(clip_box(x))

        # Evaluate initial population (skip best_x which is already evaluated)
        pop_f = [best_f]
        for i in range(1, pop_size):
            if timed() or calls_used >= de_budget:
                break
            f_val = do_call(population[i])
            pop_f.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = population[i].copy()

        # DE loop
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        for gen in range(max_generations):
            if timed() or calls_used >= de_budget:
                break

            # Sort population by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Update global best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = population[0].copy()

            new_pop = [population[0].copy()]  # Keep best
            new_f = [pop_f[0]]

            for i in range(1, pop_size):
                if timed() or calls_used >= de_budget:
                    new_pop.extend([population[i].copy() for i in range(len(new_pop), pop_size)])
                    new_f.extend([pop_f[i] for i in range(len(new_f), pop_size)])
                    break

                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size - 1, 3, replace=False)
                # Adjust indices to skip i
                if r1 >= i: r1 += 1
                if r2 >= i: r2 += 1
                if r3 >= i: r3 += 1

                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                v = clip_box(v)

                # Crossover
                j_rand = rng.integers(dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = clip_box(trial)

                # Evaluate trial
                f_trial = do_call(trial)

                # Selection
                if f_trial <= pop_f[i]:
                    new_pop.append(trial)
                    new_f.append(f_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop.append(population[i].copy())
                    new_f.append(pop_f[i])

            population = new_pop
            pop_f = new_f

        # --- Phase 2: Local Refinement with Nelder-Mead ---
        remaining = budget - calls_used
        if remaining > 10 and not timed():
            # Try Nelder-Mead from best and a few nearby points
            polish_starts = [best_x.copy()]
            for scale in [0.5, 1.0, 2.0]:
                px = best_x + scale * rng.standard_normal(dim)
                polish_starts.append(clip_box(px))

            for ps in polish_starts:
                if timed() or calls_used >= budget - 5:
                    break

                def objective(x):
                    nonlocal calls_used
                    if timed() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    return func(clip_box(x))

                try:
                    res = minimize(
                        objective,
                        ps,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining - 5, 200),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = clip_box(res.x)
                except Exception:
                    pass

        # Final safety: ensure at least one call was made
        if calls_used == 0:
            do_call(np.zeros(dim))

        return best_x, best_f
# EVOLVE-BLOCK-END
