"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with local polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        try:
            # Phase 1: Differential Evolution
            pop_size = 16
            F = 0.8  # mutation scale
            CR = 0.9  # crossover rate
            max_generations = 25

            # Initialize population
            pop = rng.uniform(-5, 5, (pop_size, dim))
            pop_f = np.array([evaluate(pop[i]) for i in range(pop_size)])
            best_idx = np.argmin(pop_f)

            for gen in range(max_generations):
                if calls_made >= budget or time.time() >= deadline:
                    break

                for i in range(pop_size):
                    if calls_made >= budget or time.time() >= deadline:
                        break

                    # Pick three distinct individuals
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    while r1 == r2 or r1 == r3 or r2 == r3:
                        r1, r2, r3 = rng.choice(pop_size, 3, replace=False)

                    # Mutation
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])

                    # Crossover
                    trial = pop[i].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]

                    # Evaluate trial
                    trial_f = evaluate(trial)

                    # Selection
                    if trial_f < pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = trial_f

                # Track best
                gen_best_idx = np.argmin(pop_f)
                if pop_f[gen_best_idx] < best_f:
                    best_f = pop_f[gen_best_idx]
                    best_x = pop[gen_best_idx].copy()

            # Phase 2: Local polish around best_x using (1+1)-ES
            if calls_made < budget and time.time() < deadline:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5
                no_improve = 0
                max_no_improve = 30

                while calls_made < budget and time.time() < deadline:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = min(sigma * 1.1, 2.0)
                        no_improve = 0
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma = max(sigma * 0.85, 0.001)
                        no_improve += 1

                    if no_improve >= max_no_improve:
                        # Restart from best with smaller sigma
                        x = best_x + rng.normal(0, 0.1, dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls_made += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma = 0.3
                        no_improve = 0

            # Phase 3: Final fine polish
            if calls_made < budget and time.time() < deadline:
                px = best_x.copy()
                pfx = best_f
                polish_sigma = 0.05
                no_polish_improve = 0
                max_polish_no_improve = 15

                while calls_made < budget and time.time() < deadline:
                    py = np.clip(px + polish_sigma * rng.standard_normal(dim), -5, 5)
                    pfy = func(py)
                    calls_made += 1
                    if pfy < pfx:
                        px = py
                        pfx = pfy
                        polish_sigma = min(polish_sigma * 1.05, 0.5)
                        no_polish_improve = 0
                        if pfx < best_f:
                            best_f = pfx
                            best_x = px.copy()
                    else:
                        polish_sigma = max(polish_sigma * 0.90, 0.0005)
                        no_polish_improve += 1
                    if no_polish_improve >= max_polish_no_improve:
                        break

        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
