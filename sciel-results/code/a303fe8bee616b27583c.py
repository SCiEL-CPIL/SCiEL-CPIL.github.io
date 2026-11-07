"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution + Nelder-Mead hybrid for rotated Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(d)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution for global search
        # Population size: small for 5D, ~10-15
        pop_size = max(10, min(15, budget // 30))
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Initialize population with diverse points
        pop = np.zeros((pop_size, d))
        for i in range(pop_size):
            if i == 0:
                pop[i] = np.zeros(d)
            else:
                pop[i] = rng.uniform(-5, 5, d)

        fvals = np.array([evaluate(pop[i]) for i in range(pop_size)])
        order = np.argsort(fvals)
        pop = pop[order]
        fvals = fvals[order]

        de_calls = pop_size
        stagnation_count = 0
        best_f_de = best_f

        # DE loop
        while calls < budget - 50:  # reserve 50 for local polish
            if time.time() > deadline:
                break

            # Check for stagnation
            if best_f_de == best_f:
                stagnation_count += 1
            else:
                best_f_de = best_f
                stagnation_count = 0

            # If stagnating, try a restart with perturbation
            if stagnation_count > 3 and calls < budget - 100:
                # Perturb the best solution and restart DE locally
                pop[0] = np.clip(best_x + rng.standard_normal(d) * 0.5, -5, 5)
                fvals[0] = evaluate(pop[0])
                de_calls += 1
                # Also add some random diversity
                for i in range(1, min(3, pop_size)):
                    pop[i] = rng.uniform(-5, 5, d)
                    fvals[i] = evaluate(pop[i])
                    de_calls += 1
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                stagnation_count = 0
                continue

            # Standard DE iteration
            for i in range(pop_size):
                if calls >= budget - 50:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct individuals different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if len(set([i, r1, r2, r3])) == 4:
                        break

                # Mutation: DE/rand/1
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, -5, 5)

                # Crossover: binomial
                trial = pop[i].copy()
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
                f_trial = evaluate(trial)
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial

                # Keep sorted
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

        # Phase 2: Nelder-Mead local polish from the best point
        if calls < budget and time.time() < deadline:
            self._nelder_mead(func, best_x.copy(), best_f, d, budget - calls, deadline)

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, d, remaining_budget, deadline):
        """Nelder-Mead simplex method for local refinement."""
        nonlocal calls, best_x, best_f

        # Initialize simplex
        alpha = 1.0  # initial step
        simplex = np.zeros((d + 1, d))
        simplex[0] = x0
        fvals = np.zeros(d + 1)
        fvals[0] = f0

        for i in range(1, d + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += alpha
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            calls += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

        # NM parameters
        alpha_nm = 1.0  # reflection
        gamma_nm = 2.0  # expansion
        rho_nm = 0.5    # contraction
        sigma_nm = 0.5  # shrink

        max_iters = min(remaining_budget - 1, 200)
        for _ in range(max_iters):
            if calls >= self.budget or time.time() > deadline:
                break

            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:d], axis=0)

            # Reflection
            x_r = centroid + alpha_nm * (centroid - simplex[d])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma_nm * (centroid - simplex[d])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                if f_e < f_r:
                    simplex[d] = x_e
                    fvals[d] = f_e
                else:
                    simplex[d] = x_r
                    fvals[d] = f_r
            elif f_r < fvals[d - 1]:
                # Accept reflection
                simplex[d] = x_r
                fvals[d] = f_r
            else:
                # Contraction
                if f_r < fvals[d]:
                    # Outer contraction
                    x_c = centroid + rho_nm * (simplex[d] - centroid)
                else:
                    # Inner contraction
                    x_c = centroid - rho_nm * (simplex[d] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()

                if f_c < min(f_r, fvals[d]):
                    simplex[d] = x_c
                    fvals[d] = f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    if calls >= self.budget:
                        break
# EVOLVE-BLOCK-END
