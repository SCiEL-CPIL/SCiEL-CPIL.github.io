import numpy as np
import time
import math

class Optimizer:
    """Differential Evolution based optimizer for Katsuura function f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        evals_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = clip(x)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Structured initialization
        # Evaluate center and a set of diverse points
        x_center = np.zeros(dim)
        evaluate(x_center)

        # Evaluate vertices of [-4, 4]^dim
        for i in range(2 ** dim):
            if time.time() > deadline or evals_used >= budget:
                break
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 8 - 4
            evaluate(v)

        # Add some random points in [-4, 4]^dim
        n_init = 50
        for i in range(n_init):
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            evaluate(x)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 2: Differential Evolution
        # Population size
        pop_size = 30
        # Create initial population from best points and random perturbations
        population = [best_x.copy() for _ in range(pop_size)]
        for i in range(1, pop_size):
            population[i] = best_x + rng.normal(0, 1.5, dim)
            population[i] = clip(population[i])

        # Evaluate initial population
        pop_f = np.zeros(pop_size)
        for i in range(pop_size):
            if time.time() > deadline or evals_used >= budget:
                break
            pop_f[i] = evaluate(population[i])

        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        # Run DE until budget or time is exhausted
        stagnation_counter = 0
        max_stagnation = 200

        while time.time() < deadline and evals_used < budget:
            # Find best in population
            best_idx = np.argmin(pop_f)
            current_best = pop_f[best_idx]

            if current_best < best_f - 1e-10:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max_stagnation:
                # Restart with new diversity
                for i in range(pop_size):
                    if time.time() > deadline or evals_used >= budget:
                        break
                    if i == 0:
                        population[i] = best_x.copy()
                        pop_f[i] = best_f
                    else:
                        population[i] = best_x + rng.normal(0, 2.0, dim)
                        population[i] = clip(population[i])
                        pop_f[i] = evaluate(population[i])
                stagnation_counter = 0
                continue

            # Generate offspring for each member
            for i in range(pop_size):
                if time.time() > deadline or evals_used >= budget:
                    break

                # Select three distinct random indices
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != b and b != c and a != c and a != i and b != i and c != i:
                        break

                # Mutation
                mutant = population[a] + F * (population[b] - population[c])

                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = clip(trial)
                f_trial = evaluate(trial)

                # Selection
                if f_trial <= pop_f[i]:
                    population[i] = trial
                    pop_f[i] = f_trial

        # Phase 3: Local refinement with Nelder-Mead from best point
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline:
            def nelder_mead(x0, max_evals, step=0.5):
                nonlocal best_f, best_x, evals_used
                n = dim
                simplex = np.zeros((n + 1, n))
                simplex[0] = x0
                for i in range(n):
                    simplex[i + 1] = x0.copy()
                    simplex[i + 1][i] += step

                fvals = np.zeros(n + 1)
                for i in range(n + 1):
                    if time.time() > deadline or evals_used >= budget:
                        return
                    fvals[i] = evaluate(simplex[i])

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                for _ in range(max_evals):
                    if time.time() > deadline or evals_used >= budget:
                        return

                    idx = np.argsort(fvals)
                    simplex = simplex[idx]
                    fvals = fvals[idx]

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + alpha * (centroid - simplex[-1])
                    f_r = evaluate(x_r)

                    if f_r < fvals[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        f_e = evaluate(x_e)
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
                        if f_r < fvals[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        f_c = evaluate(x_c)
                        if f_c < min(f_r, fvals[-1]):
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if time.time() > deadline or evals_used >= budget:
                                    return
                                fvals[i] = evaluate(simplex[i])
                            break

            nm_budget = min(150, remaining // 2)
            nelder_mead(best_x.copy(), nm_budget, step=0.3)

        return best_x, best_f
