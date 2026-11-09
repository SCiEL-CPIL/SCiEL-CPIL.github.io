"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution + local refinement for Katsuura (f23) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if check_deadline():
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = float(func(x_clipped))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Differential Evolution ---
        # Population size for DE
        pop_size = 20
        # DE parameters
        F = 0.8  # mutation scale
        CR = 0.5  # crossover rate

        # Initialize population: dense near origin + spread
        pop = np.zeros((pop_size, dim))
        fvals = np.zeros(pop_size)

        # First few points: dense near origin (Katsuura global min tends to be near 0)
        for i in range(min(6, pop_size)):
            if i == 0:
                pop[i] = np.zeros(dim)
            else:
                pop[i] = rng.normal(0, 0.8, dim)
        # Remaining: spread across the box
        for i in range(6, pop_size):
            pop[i] = rng.uniform(-4, 4, dim)

        # Evaluate initial population
        for i in range(pop_size):
            if calls_made >= budget or check_deadline():
                break
            fvals[i] = evaluate(pop[i])
            if np.isinf(fvals[i]):
                fvals[i] = 1e30

        # Track best
        best_idx = np.argmin(fvals)
        if best_x is not None:
            best_f = fvals[best_idx]
            best_x = pop[best_idx].copy()

        # Stagnation tracking
        no_improve = 0
        stagnation_threshold = 15  # generations without improvement

        # Adaptive parameters
        F_current = F
        CR_current = CR

        # DE main loop
        while calls_made + pop_size <= budget and not check_deadline():
            # Generate offspring
            offspring = np.zeros((pop_size, dim))
            off_fvals = np.zeros(pop_size)

            for i in range(pop_size):
                if calls_made >= budget or check_deadline():
                    break
                # Select three distinct random indices different from i
                while True:
                    r1 = rng.integers(pop_size)
                    r2 = rng.integers(pop_size)
                    r3 = rng.integers(pop_size)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break

                # Mutation: DE/rand/1
                mutant = pop[r1] + F_current * (pop[r2] - pop[r3])

                # Crossover: binomial
                j_rand = rng.integers(dim)
                cross = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR_current or j == j_rand:
                        cross[j] = mutant[j]
                    else:
                        cross[j] = pop[i][j]

                offspring[i] = np.clip(cross, -5.0, 5.0)
                off_fvals[i] = evaluate(offspring[i])
                if np.isinf(off_fvals[i]):
                    off_fvals[i] = 1e30

            # Selection: keep better of parent vs offspring
            improved = False
            for i in range(pop_size):
                if off_fvals[i] < fvals[i]:
                    pop[i] = offspring[i]
                    fvals[i] = off_fvals[i]
                    improved = True

            # Update global best
            local_best_idx = np.argmin(fvals)
            if fvals[local_best_idx] < best_f - 1e-10:
                best_f = fvals[local_best_idx]
                best_x = pop[local_best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Adaptive parameter adjustment
            if no_improve > stagnation_threshold:
                # Perturb best individual and reset stagnation counter
                if best_x is not None:
                    perturbed = best_x + rng.normal(0, 0.5, dim)
                    perturbed = np.clip(perturbed, -5.0, 5.0)
                    f_pert = evaluate(perturbed)
                    if not np.isinf(f_pert) and f_pert < best_f:
                        best_f = f_pert
                        best_x = perturbed.copy()
                        no_improve = 0
                    else:
                        no_improve = 0  # reset to avoid infinite loop

                # Adapt F and CR
                F_current = np.clip(F_current + rng.uniform(-0.2, 0.2), 0.4, 1.2)
                CR_current = np.clip(CR_current + rng.uniform(-0.1, 0.1), 0.3, 0.7)

            # Check convergence: if population diversity is very low, inject diversity
            if no_improve > 30:
                # Inject a few random individuals
                for i in range(3):
                    idx = rng.integers(pop_size)
                    pop[idx] = rng.uniform(-4, 4, dim)
                    fvals[idx] = evaluate(pop[idx])
                    if np.isinf(fvals[idx]):
                        fvals[idx] = 1e30
                no_improve = 0

        # --- Phase 2: Local refinement using Nelder-Mead ---
        if best_x is not None and calls_made < budget and not check_deadline():
            # Use scipy's Nelder-Mead for local refinement
            from scipy.optimize import minimize

            # Set up Nelder-Mead
            # Initial simplex: small perturbations around best_x
            x0 = best_x.copy()
            f0 = best_f

            # Create initial simplex
            simplex = [x0.copy()]
            for j in range(dim):
                pert = x0.copy()
                pert[j] += 0.3
                simplex.append(pert)

            # Evaluate simplex
            simplex_f = [f0]
            for i in range(1, dim + 1):
                if calls_made >= budget or check_deadline():
                    break
                simplex_f.append(evaluate(simplex[i]))

            # Run Nelder-Mead manually (to control budget)
            n_iter_max = min(100, (budget - calls_made) // (dim + 1))
            for _ in range(n_iter_max):
                if calls_made >= budget or check_deadline():
                    break

                # Sort by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = evaluate(x_ref)
                if np.isinf(f_ref):
                    f_ref = 1e30

                if f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = evaluate(x_exp)
                    if np.isinf(f_exp):
                        f_exp = 1e30
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = evaluate(x_con)
                    if np.isinf(f_con):
                        f_con = 1e30
                    if f_con < worst_f:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
                            if np.isinf(simplex_f[i]):
                                simplex_f[i] = 1e30

                # Update best
                local_best_idx = np.argmin(simplex_f)
                if simplex_f[local_best_idx] < best_f - 1e-10:
                    best_f = simplex_f[local_best_idx]
                    best_x = simplex[local_best_idx].copy()

        # Return the best solution found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
