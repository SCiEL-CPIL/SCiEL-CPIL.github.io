"""Multi-objective optimizer on WFG7 using DE/rand/1 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE/rand/1 multi-objective optimizer with non-dominated set maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng()

        # --- Structured initialization: Latin Hypercube over population ---
        pop_size = 16
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(pop_size)
            for i in range(pop_size):
                X[i, d] = lo[d] + (hi[d] - lo[d]) * (perm[i] + rng.random()) / pop_size
        X = np.clip(X, lo, hi)

        # Evaluate initial population
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = pop_size

        # --- Maintain non-dominated set (vectors only, for hypervolume) ---
        nd_F = F.copy()  # non-dominated objective vectors

        def is_dominated(f, nd_F_arr):
            """Return True if f is dominated by any row of nd_F_arr."""
            if nd_F_arr.shape[0] == 0:
                return False
            # f dominated by g if all g <= f and at least one g < f
            return np.any(np.all(nd_F_arr <= f, axis=1) & np.any(nd_F_arr < f, axis=1))

        def dominates(f, nd_F_arr):
            """Return boolean mask of rows in nd_F_arr dominated by f."""
            if nd_F_arr.shape[0] == 0:
                return np.array([])
            return np.all(f <= nd_F_arr, axis=1) & np.any(f < nd_F_arr, axis=1)

        # --- Differential Evolution main loop ---
        F_scale = 0.8
        crossover_rate = 0.5

        while used < self.budget:
            if time.time() > deadline:
                break

            # Select three distinct parents
            i = rng.integers(pop_size)
            others = list(range(pop_size))
            others.remove(i)
            a = rng.choice(others)
            rest = [x for x in others if x != a]
            b = rng.choice(rest)

            # DE/rand/1 mutation
            trial = X[a] + F_scale * (X[b] - X[i])
            # Crossover (binomial)
            jrand = rng.integers(self.dim)
            mask = rng.random(self.dim) < crossover_rate
            mask[jrand] = True
            trial = np.where(mask, trial, X[i])
            trial = np.clip(trial, lo, hi)

            f_trial = np.asarray(problem(trial), dtype=float)
            used += 1

            # Replacement: greedy - replace parent i if trial is not dominated by F[i]
            if not is_dominated(f_trial, F[i].reshape(1, -1)):
                X[i] = trial
                F[i] = f_trial

            # Update non-dominated set
            if not is_dominated(f_trial, nd_F):
                # Remove points dominated by f_trial
                dom_mask = dominates(f_trial, nd_F)
                nd_F = nd_F[~dom_mask]
                nd_F = np.vstack([nd_F, f_trial])
            # Also re-check: some existing non-dominated points may now be dominated by the new F
            # (only if F changed, but greedy keeps all evals; nd_F is built from all evals)
            # We already add f_trial if non-dominated. The old nd_F entries remain valid
            # because adding a new non-dominated point cannot dominate existing non-dominated points
            # (they were mutually non-dominated before; f_trial is non-dominated by them).

        return X
# EVOLVE-BLOCK-END
