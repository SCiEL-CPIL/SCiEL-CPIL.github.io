"""Structural change: Differential Evolution + local refinement for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE with adaptive parameters and local refinement for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin

        dim = self.dim
        budget = self.budget

        best_x = np.full(dim, 0.0)
        best_f = float('inf')
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Differential Evolution
        def de_phase(budget_de, rng):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return

            pop_size = 12
            F = 0.8  # mutation factor
            CR = 0.9  # crossover rate

            # Initialize population
            pop = [rng.uniform(-5, 5, dim) for _ in range(pop_size)]
            fit = [safe_call(x) for x in pop]

            # Track best in population
            pop_best_idx = np.argmin(fit)
            pop_best_x = pop[pop_best_idx].copy()
            pop_best_f = fit[pop_best_idx]

            # Stagnation tracking
            stagnation_count = 0
            improvement_threshold = 1e-10

            for gen in range(budget_de // pop_size + 1):
                if call_count >= budget or time.time() > deadline:
                    break

                # Check stagnation
                if pop_best_f - best_f < improvement_threshold and stagnation_count > 15:
                    break

                new_pop = []
                new_fit = []

                for i in range(pop_size):
                    if call_count >= budget or time.time() > deadline:
                        break

                    # Select 3 distinct individuals different from i
                    while True:
                        r1 = rng.integers(0, pop_size)
                        r2 = rng.integers(0, pop_size)
                        r3 = rng.integers(0, pop_size)
                        if r1 != i and r2 != i and r3 != i and r1 != r2 and r1 != r3 and r2 != r3:
                            break

                    # Mutation: DE/rand/1
                    v = pop[r1] + F * (pop[r2] - pop[r3])

                    # Crossover: binomial
                    j_rand = rng.integers(0, dim)
                    u = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]

                    u = np.clip(u, -5.0, 5.0)
                    fu = safe_call(u)

                    # Selection
                    if fu <= fit[i]:
                        new_pop.append(u)
                        new_fit.append(fu)
                    else:
                        new_pop.append(pop[i])
                        new_fit.append(fit[i])

                if call_count >= budget or time.time() > deadline:
                    break

                pop = new_pop
                fit = new_fit

                # Update population best
                new_best_idx = np.argmin(fit)
                if fit[new_best_idx] < pop_best_f - improvement_threshold:
                    pop_best_f = fit[new_best_idx]
                    pop_best_x = pop[new_best_idx].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1

            return

        # Phase 2: Local refinement using Nelder-Mead-like simplex
        def local_refine(x_start, budget_local, rng):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = [x_start.copy()]
            for j in range(dim):
                point = x_start.copy()
                step = 0.1 if abs(x_start[j]) > 1e-10 else 0.1
                point[j] += step
                point = np.clip(point, -5.0, 5.0)
                simplex.append(point)

            simplex_fit = [safe_call(x) for x in simplex]

            for _ in range(budget_local // (dim + 1) + 1):
                if call_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_fit)
                simplex = [simplex[i] for i in order]
                simplex_fit = [simplex_fit[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)

                if fr < simplex_fit[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_fit[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_fit[-1] = fr
                elif fr < simplex_fit[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_fit[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_call(xc)
                    if fc < simplex_fit[-1]:
                        simplex[-1] = xc
                        simplex_fit[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_fit[i] = safe_call(simplex[i])

                if call_count >= budget or time.time() > deadline:
                    break

            return

        # Main algorithm
        rng = np.random.default_rng(123)

        # Phase 1: DE with 70% of budget
        budget_de = int(budget * 0.7)
        de_phase(budget_de, rng)

        # Phase 2: Local refinement with remaining budget
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline:
            local_refine(best_x.copy(), remaining, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
