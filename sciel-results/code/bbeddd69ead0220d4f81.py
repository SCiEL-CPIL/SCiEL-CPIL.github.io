"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + Differential Evolution + Polish phase for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        d = self.dim

        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = self.budget

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start to find a good initial basin
        starts = [np.zeros(d)]
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, d))

        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = s_clipped.copy()

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: Differential Evolution
        # Population size: small to fit budget
        pop_size = 10
        # Initialize population: one is best, rest are diverse
        pop = np.zeros((pop_size, d))
        pop[0] = best_x
        for i in range(1, pop_size):
            if i % 2 == 1:
                # Local: around best
                pop[i] = np.clip(best_x + 1.5 * rng.standard_normal(d), -5, 5)
            else:
                # Global: random
                pop[i] = rng.uniform(-5, 5, d)

        # Evaluate initial population
        pop_f = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            fx = try_eval(pop[i])
            if fx is None:
                break
            pop_f[i] = fx

        # DE parameters
        F = 0.8
        CR = 0.9
        stagnation = 0
        max_stagnation = 20

        while calls_used < max_calls:
            if time.time() > deadline - 5:
                break

            # Find best in population
            best_idx = np.argmin(pop_f)
            current_best_f = pop_f[best_idx]

            improved_this_gen = False

            # For each individual, generate a trial
            for i in range(pop_size):
                if calls_used >= max_calls:
                    break
                if time.time() > deadline - 3:
                    break

                # Select 3 distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                trial = pop[i].copy()
                for k in range(d):
                    if rng.random() < CR or k == rng.integers(d):
                        trial[k] = mutant[k]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                fy = try_eval(trial)
                if fy is None:
                    break

                # Selection
                if fy < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = fy
                    if fy < current_best_f:
                        current_best_f = fy
                        improved_this_gen = True

            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1

            # If stuck, inject diversity
            if stagnation >= max_stagnation:
                # Replace worst half with random points
                worst_half = np.argsort(pop_f)[pop_size // 2:]
                for idx in worst_half:
                    pop[idx] = rng.uniform(-5, 5, d)
                    fx = try_eval(pop[idx])
                    if fx is None:
                        break
                    pop_f[idx] = fx
                stagnation = 0

        # Phase 3: Polish phase - use very small sigma for fine-tuning
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            polish_calls = min(30, max_calls - calls_used)
            for _ in range(polish_calls):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.7  # shrink further on success
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
