"""Rewritten optimizer: Differential Evolution with adaptive F + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """DE/rand/1 with adaptive mutation factor + final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under 300s
        dim = self.dim
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        calls = [0]
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            calls[0] += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, budget_remaining):
            """Nelder-Mead local refinement from x0 for budget_remaining evaluations."""
            nonlocal calls, best_f, best_x
            x = np.array(x0, dtype=float)
            f0 = counted(x)

            # Build simplex
            simplex = np.empty((dim + 1, dim))
            simplex[0] = x
            fvals = np.empty(dim + 1)
            fvals[0] = f0

            for i in range(dim):
                p = x.copy()
                p[i] += 0.1
                p = np.clip(p, lo, hi)
                simplex[i + 1] = p
                fvals[i + 1] = counted(p)

            alpha, gamma, rho = 1.0, 2.0, 0.5

            while calls[0] < budget_remaining and time.time() < deadline:
                # Sort simplex by fvals
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                x_best = simplex[0]
                x_worst = simplex[-1]
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, lo, hi)
                f_r = counted(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = counted(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = counted(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            fvals[i] = counted(simplex[i])

                # Check convergence
                if np.std(fvals) < 1e-10:
                    break

            return simplex[0], fvals[0]

        # Phase 1: Differential Evolution
        # Population size: 10 (5x dim)
        pop_size = max(10, 2 * dim)
        de_budget = int(self.budget * 0.7)

        # Initialize population
        pop = np.zeros((pop_size, dim))
        fvals = np.zeros(pop_size)
        for i in range(pop_size):
            if i == 0:
                pop[i] = np.zeros(dim)
            else:
                pop[i] = rng.uniform(lo, hi, dim)
            pop[i] = np.clip(pop[i], lo, hi)
            fvals[i] = counted(pop[i])

        # Adaptive mutation factor
        F = 0.8
        F_min, F_max = 0.3, 0.9
        CR = 0.7  # crossover rate

        # Track best
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_f:
            best_f = fvals[best_idx]
            best_x = pop[best_idx].copy()

        # DE loop
        while calls[0] < de_budget and time.time() < deadline:
            # Update F adaptively: reduce if no improvement in last few generations
            # Simple heuristic: decrease F slightly each generation, reset if improvement
            improved = False
            for i in range(pop_size):
                if calls[0] >= de_budget or time.time() >= deadline:
                    break

                # Select three distinct random individuals
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                candidates = np.where(mask)[0]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, lo, hi)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, lo, hi)
                f_trial = counted(trial)

                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    improved = True
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        # Reset F to higher value on improvement
                        F = min(F + 0.1, F_max)

            if not improved:
                # No improvement this generation, reduce F
                F = max(F - 0.05, F_min)

            # Also update global best from population
            pop_best_idx = np.argmin(fvals)
            if fvals[pop_best_idx] < best_f:
                best_f = fvals[pop_best_idx]
                best_x = pop[pop_best_idx].copy()

        # Phase 2: Local refinement with Nelder-Mead from best point
        if calls[0] < self.budget and time.time() < deadline:
            refine_budget = self.budget - calls[0]
            if best_x is not None:
                nelder_mead(best_x, refine_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
