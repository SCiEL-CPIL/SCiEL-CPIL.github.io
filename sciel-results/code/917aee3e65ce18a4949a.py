"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Time guard: return best if we're running out of time
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        # Differential Evolution parameters
        pop_size = 20  # population size
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate

        # Initialize population uniformly in [-5, 5]^n
        pop = rng.uniform(-5, 5, (pop_size, n))
        fvals = np.array([func(x) for x in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Track best-so-far
        global_best_x = best_x.copy()
        global_best_f = best_f

        # Adaptive parameters: start with larger F, CR for exploration
        # Then shift to exploitation

        calls_used = pop_size

        # Main DE loop
        gen = 0
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            gen += 1
            # Adapt F and CR: reduce over generations for exploitation
            progress = min(1.0, gen / 50.0)
            F_adapt = F * (1.0 - 0.3 * progress)  # reduce differential weight
            CR_adapt = CR * (1.0 - 0.1 * progress)  # reduce crossover rate

            # Update population
            new_pop = np.empty_like(pop)
            new_fvals = np.empty(pop_size)

            for i in range(pop_size):
                # Check time
                if time.time() > deadline:
                    break

                # Select 3 distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if len(set([r1, r2, r3, i])) == 4:
                        break

                # Create trial vector using DE/rand/1
                v = pop[r1] + F_adapt * (pop[r2] - pop[r3])

                # Binary crossover
                j_rand = rng.integers(n)
                trial = np.empty(n)
                for j in range(n):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ft = func(trial)
                calls_used += 1

                # Selection
                if ft <= fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = ft
                else:
                    new_pop[i] = pop[i]
                    new_fvals[i] = fvals[i]

            # Update population
            pop = new_pop
            fvals = new_fvals

            # Update best
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]

            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        # Local refinement using Nelder-Mead on the best point
        if calls_used < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func,
                    global_best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    }
                )
                if result.fun < global_best_f:
                    global_best_x = result.x.copy()
                    global_best_f = result.fun
            except Exception:
                pass

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
