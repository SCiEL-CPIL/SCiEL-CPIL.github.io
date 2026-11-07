"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE + multi-start local search for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: evaluator kills after 330s. Use 310s for safety.
        start_time = time.time()
        max_time = 310.0

        def time_ok():
            return time.time() - start_time < max_time

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution for global exploration
        N = 12  # Population size
        F = 0.8
        CR = 0.9

        # Initialize population with spread coverage
        pop = rng.uniform(-5, 5, (N, dim))
        fit = np.array([func(x) for x in pop])
        calls = N

        best_idx = np.argmin(fit)
        x_best = pop[best_idx].copy()
        f_best = fit[best_idx]

        # Archive for restarts
        archive = [(f_best, x_best.copy())]

        # DE Loop
        while calls + N <= budget and time_ok():
            new_pop = pop.copy()
            new_fit = fit.copy()

            for i in range(N):
                # Select 3 distinct random indices different from i
                perm = rng.permutation(N)
                mask = perm != i
                a_idx, b_idx, c_idx = perm[mask][:3]

                # Mutation: DE/rand/1/bin
                v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])

                # Crossover
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        new_pop[i][j] = v[j]
                    else:
                        new_pop[i][j] = pop[i][j]

                # Clip
                new_pop[i] = np.clip(new_pop[i], -5, 5)

                # Evaluate
                f_new = func(new_pop[i])
                calls += 1

                if not time_ok():
                    break

                # Selection
                if f_new <= new_fit[i]:
                    new_fit[i] = f_new
                    if f_new < f_best:
                        f_best = f_new
                        x_best = new_pop[i].copy()
                        archive.append((f_best, x_best.copy()))

            if not time_ok():
                break

            pop = new_pop
            fit = new_fit

            # Stall detection: replace worst half if population collapsed
            if len(np.unique(fit)) < N // 2:
                worst_indices = np.argsort(fit)[-N // 2:]
                for idx in worst_indices:
                    pop[idx] = rng.uniform(-5, 5, dim)
                    fit[idx] = func(pop[idx])
                    calls += 1
                    if fit[idx] < f_best:
                        f_best = fit[idx]
                        x_best = pop[idx].copy()
                        archive.append((f_best, x_best.copy()))
                    if not time_ok():
                        break

        # Phase 2: Multi-start local refinement using top candidates
        # Sort archive by fitness and take top candidates
        archive.sort(key=lambda x: x[0])
        top_candidates = [x for _, x in archive[:min(5, len(archive))]]

        # For each top candidate, do (1+1) CMA-ES-like refinement
        remaining_budget = budget - calls
        if remaining_budget > 10 and time_ok():
            # Split remaining budget among candidates
            num_refine = min(len(top_candidates), max(1, remaining_budget // 30))
            ref_budget_each = max(5, remaining_budget // num_refine)

            for cand in top_candidates[:num_refine]:
                if not time_ok():
                    break

                x_c = cand.copy()
                f_c = func(x_c)
                calls += 1

                if f_c < f_best:
                    f_best = f_c
                    x_best = x_c.copy()

                sigma = 1.0
                for step in range(ref_budget_each):
                    if not time_ok() or calls >= budget:
                        break

                    # Generate trial point
                    z = rng.standard_normal(dim)
                    x_trial = x_c + sigma * z
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < f_c:
                        x_c = x_trial
                        f_c = f_trial
                        if f_c < f_best:
                            f_best = f_c
                            x_best = x_c.copy()
                        sigma *= 1.1  # Expand on success
                    else:
                        sigma *= 0.8  # Contract on failure
                    sigma = np.clip(sigma, 1e-4, 5.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
