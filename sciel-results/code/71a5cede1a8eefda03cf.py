"""Multi-objective optimizer on WFG7 using MODE with adaptive scaling and time guard."""
import numpy as np
import time


class Optimizer:
    """Multi-Objective Differential Evolution (MODE) for WFG7 with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: balance diversity vs. budget
        N = min(50, max(20, self.budget // 4))
        N = min(N, self.budget)

        # Initialize population
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Adaptive scaling factor
        F_mut = 0.8
        CR = 0.9

        # DE/rand/1 with binomial crossover
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Pick base individual
            i = rng.integers(N)

            # Pick three distinct random individuals
            perm = rng.permutation(N)
            a, b, c = perm[0], perm[1], perm[2]

            # Mutation
            v = X[a] + F_mut * (X[b] - X[c])

            # Clip to bounds
            v = np.clip(v, lo, hi)

            # Binomial crossover
            trial = X[i].copy()
            j_rand = rng.integers(self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]

            # Evaluate trial
            f_trial = problem(trial)
            used += 1

            # Selection: keep the better of parent and trial
            # Parent i dominates trial -> keep parent
            # Trial dominates parent -> replace
            # Otherwise -> replace (or keep based on crowding, but simple replacement works for DE)
            if not (np.all(f_trial <= F[i]) and np.any(f_trial < F[i])):
                # Trial does NOT dominate parent
                if not (np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)):
                    # Incomparable: replace to maintain diversity
                    X[i] = trial
                    F[i] = f_trial
                # If parent dominates trial, keep parent
            else:
                # Trial dominates parent
                X[i] = trial
                F[i] = f_trial

            # Adaptive mutation factor: decrease slightly over time
            F_mut = max(0.5, F_mut * 0.999)

        return X
