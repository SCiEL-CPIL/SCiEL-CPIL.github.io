import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Differential Evolution with adaptive parameters
        pop_size = 12
        max_iter = (budget - pop_size) // pop_size

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        calls_made = pop_size

        # Adaptive DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Track stagnation
        stagnation_count = 0
        prev_best_f = best_f

        for gen in range(max_iter):
            # Check time
            if time.time() > deadline:
                break

            gen_best_f = best_f

            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Select 3 distinct random indices different from i
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if a != i and b != i and c != i:
                        break

                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                u = v.copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]

                u = np.clip(u, -5, 5)
                fu = func(u)
                calls_made += 1

                if fu < fpop[i]:
                    pop[i] = u
                    fpop[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()

            # Check for stagnation
            if best_f >= prev_best_f - 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best_f = best_f

            # Adapt F and CR based on progress
            if gen % 10 == 0 and gen > 0:
                if stagnation_count > 2:
                    # Stagnating: increase exploration
                    F = np.clip(F + 0.1, 0.4, 1.0)
                    CR = np.clip(CR + 0.05, 0.5, 0.95)
                    # Perturb population slightly to escape local minima
                    for i in range(pop_size):
                        pop[i] = np.clip(pop[i] + 0.1 * rng.standard_normal(dim), -5, 5)
                        # Re-evaluate perturbed individuals
                        fpop[i] = func(pop[i])
                        calls_made += 1
                        if fpop[i] < best_f:
                            best_f = fpop[i]
                            best_x = pop[i].copy()
                else:
                    # Making progress: fine-tune
                    F = np.clip(F + rng.uniform(-0.05, 0.05), 0.4, 1.0)
                    CR = np.clip(CR + rng.uniform(-0.03, 0.03), 0.5, 0.95)

        # Final polish: (1+1)-ES with adaptive step size
        if time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            # Estimate remaining budget
            remaining = budget - calls_made
            if remaining > 20:
                for step in range(min(remaining, 100)):
                    if time.time() > deadline:
                        break

                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.3
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.85

                    if sigma < 1e-6:
                        break

        return best_x, best_f
