"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: multi-restart differential evolution with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Track all evaluated points for hypervolume
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            all_X.append(x.copy())
            all_F.append(fc)
            used += 1
            return fc

        # Multi-restart DE with larger populations for better coverage
        n_restarts = 3
        pop_size = min(20, self.budget // n_restarts // 2)
        per_restart_budget = self.budget // n_restarts

        for r in range(n_restarts):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Initialize population for this restart
            if r == 0:
                X = lo + (hi - lo) * rng.random((pop_size, self.dim))
            else:
                # Initialize from best points of previous archives + random
                if len(all_F) > 0:
                    F_arr = np.array(all_F)
                    # Pick some good points (low sum) as seeds
                    sums = np.sum(F_arr, axis=1)
                    n_seeds = min(pop_size // 2, len(all_F))
                    if n_seeds > 0:
                        seed_idx = rng.choice(len(all_F), size=n_seeds, replace=False)
                        X = np.array([all_X[i].copy() for i in seed_idx])
                        remaining = pop_size - len(X)
                        if remaining > 0:
                            X = np.vstack([X, lo + (hi - lo) * rng.random((remaining, self.dim))])
                    else:
                        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
                else:
                    X = lo + (hi - lo) * rng.random((pop_size, self.dim))

            F = np.array([evaluate(x) for x in X])

            # DE loop within this restart
            max_iter = per_restart_budget - pop_size
            while used < self.budget and max_iter > 0:
                if time.time() > deadline:
                    break
                max_iter -= 1

                for i in range(pop_size):
                    if used >= self.budget or time.time() > deadline:
                        break

                    # DE/rand/1 mutation
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    while a == i or b == i or c == i:
                        a, b, c = rng.choice(pop_size, size=3, replace=False)

                    F_factor = 0.5 + 0.5 * rng.random()
                    v = X[a] + F_factor * (X[b] - X[c])
                    v = np.clip(v, lo, hi)

                    # Crossover
                    CR = 0.5
                    j_rand = rng.integers(self.dim)
                    trial = X[i].copy()
                    for j in range(self.dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]

                    ft = evaluate(trial)

                    # Selection
                    if np.all(ft <= F[i]) and np.any(ft < F[i]):
                        X[i] = trial
                        F[i] = ft
                    elif np.all(F[i] <= ft) and np.any(F[i] < ft):
                        pass  # keep parent
                    else:
                        # Non-dominated: replace if trial has lower sum
                        if np.sum(ft) < np.sum(F[i]):
                            X[i] = trial
                            F[i] = ft

        return all_X
# EVOLVE-BLOCK-END
