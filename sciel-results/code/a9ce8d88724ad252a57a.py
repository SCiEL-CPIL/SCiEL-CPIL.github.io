"""Rewritten program using Differential Evolution for robust global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def de_optimize(pop_size, F, CR, max_gens):
            """Run Differential Evolution for a given number of generations."""
            nonlocal best_x, best_f, calls_made, deadline

            # Initialize population
            pop = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
            pop_f = np.array([safe_call(ind) for ind in pop])

            # Track best
            best_idx = np.argmin(pop_f)
            local_best_x = pop[best_idx].copy()
            local_best_f = pop_f[best_idx]

            for gen in range(max_gens):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Sort population by fitness
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update local best
                if pop_f[0] < local_best_f:
                    local_best_f = pop_f[0]
                    local_best_x = pop[0].copy()

                # Generate offspring
                for i in range(pop_size):
                    if time.time() > deadline or calls_made >= budget:
                        break

                    # Select mutation strategy
                    if rng.random() < 0.5:
                        # best/1/bin
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                        while r1 == i and r2 == i:
                            r1, r2 = rng.choice(pop_size, size=2, replace=False)
                        v = pop[0] + F * (pop[r1] - pop[r2])
                    else:
                        # rand/1/bin
                        r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                        while r1 == i or r2 == i or r3 == i:
                            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                        v = pop[r1] + F * (pop[r2] - pop[r3])

                    # Clip to bounds
                    v = np.clip(v, -5.0, 5.0)

                    # Crossover (binomial)
                    j_rand = rng.integers(dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]

                    # Evaluate trial
                    trial_f = safe_call(trial)

                    # Selection
                    if trial_f <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = trial_f

                # Update local best after generation
                gen_best_idx = np.argmin(pop_f)
                if pop_f[gen_best_idx] < local_best_f:
                    local_best_f = pop_f[gen_best_idx]
                    local_best_x = pop[gen_best_idx].copy()

            return local_best_x, local_best_f

        # Phase 1: DE with moderate population for global exploration
        pop_size = max(10, 2 * dim)
        F = 0.8
        CR = 0.5
        budget_per_phase1 = budget // 2
        max_gens1 = max(1, budget_per_phase1 // pop_size)
        de_optimize(pop_size, F, CR, max_gens1)

        # Phase 2: DE with smaller population for refinement
        if best_x is not None and time.time() < deadline and calls_made < budget:
            pop_size2 = max(8, dim)
            F2 = 0.5
            CR2 = 0.7
            remaining = budget - calls_made
            max_gens2 = max(1, remaining // pop_size2)

            # Initialize around current best
            pop = np.zeros((pop_size2, dim))
            pop[0] = best_x
            for i in range(1, pop_size2):
                pop[i] = best_x + rng.normal(0, 0.5, dim)
                pop[i] = np.clip(pop[i], -5.0, 5.0)

            pop_f = np.array([safe_call(ind) for ind in pop])

            for gen in range(max_gens2):
                if time.time() > deadline or calls_made >= budget:
                    break

                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()

                for i in range(pop_size2):
                    if time.time() > deadline or calls_made >= budget:
                        break

                    if rng.random() < 0.6:
                        r1, r2 = rng.choice(pop_size2, size=2, replace=False)
                        while r1 == i and r2 == i:
                            r1, r2 = rng.choice(pop_size2, size=2, replace=False)
                        v = pop[0] + F2 * (pop[r1] - pop[r2])
                    else:
                        r1, r2, r3 = rng.choice(pop_size2, size=3, replace=False)
                        while r1 == i or r2 == i or r3 == i:
                            r1, r2, r3 = rng.choice(pop_size2, size=3, replace=False)
                        v = pop[r1] + F2 * (pop[r2] - pop[r3])

                    v = np.clip(v, -5.0, 5.0)

                    j_rand = rng.integers(dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.random() < CR2 or j == j_rand:
                            trial[j] = v[j]

                    trial_f = safe_call(trial)

                    if trial_f <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = trial_f

                gen_best_idx = np.argmin(pop_f)
                if pop_f[gen_best_idx] < best_f:
                    best_f = pop_f[gen_best_idx]
                    best_x = pop[gen_best_idx].copy()

        # Phase 3: Local search with small perturbations
        if best_x is not None and time.time() < deadline and calls_made < budget:
            sigma = 0.1
            for _ in range(min(30, budget - calls_made)):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_call(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
