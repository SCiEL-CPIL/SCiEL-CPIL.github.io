"""Optimizer for BBOB f23 (Katsuura) with multi-start DE/rand/1."""
import numpy as np
import time


class Optimizer:
    """Multi-start differential evolution for the Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size: larger for better diversity in multimodal landscape
        pop_size = min(50, max(10, budget // 10))

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])

        best_idx = int(np.argmin(fvals))
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE/rand/1 parameters tuned for Katsuura
        F = 0.5   # differential weight - lower for more exploration
        CR = 0.7  # crossover rate

        calls_made = pop_size

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate trial vectors for all population members in one generation
            # But we need to be careful about budget - generate one at a time
            # to stay within budget limits
            
            # Pick a random target
            target_idx = rng.integers(0, pop_size)
            x_target = pop[target_idx]

            # Pick two distinct random individuals different from target
            others = [i for i in range(pop_size) if i != target_idx]
            i1, i2 = rng.choice(others, size=2, replace=False)
            v = pop[i1] + F * (pop[i2] - pop[i1])
            v = np.clip(v, -5, 5)

            # Binomial crossover
            r = rng.random(dim)
            j_rand = rng.integers(0, dim)
            trial = np.where((r < CR) | (np.arange(dim) == j_rand), v, x_target)
            trial = np.clip(trial, -5, 5)

            f_trial = func(trial)
            calls_made += 1

            if f_trial <= fvals[target_idx]:
                pop[target_idx] = trial
                fvals[target_idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

            # Periodic diversity injection: restart worst individuals
            if calls_made % 30 == 0 and calls_made < budget - 10:
                # Find the worst 10% of population (excluding best)
                sorted_idx = np.argsort(fvals)
                n_worst = max(1, pop_size // 10)
                worst_idx = sorted_idx[-n_worst:]
                # Don't restart the best
                worst_idx = [i for i in worst_idx if i != best_idx]
                
                for idx in worst_idx:
                    if calls_made >= budget:
                        break
                    # Random restart with slight bias toward center
                    pop[idx] = rng.uniform(-3, 3, dim)
                    fvals[idx] = func(pop[idx])
                    calls_made += 1
                    if fvals[idx] < best_f:
                        best_f = fvals[idx]
                        best_x = pop[idx].copy()
                        best_idx = idx

        return best_x, best_f
