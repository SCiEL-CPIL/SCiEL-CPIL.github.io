"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase derivative-free optimizer for BBOB f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best
        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_eval(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start CMA-ES-like with population
        # Use a simple population-based search with adaptive step size
        rng = np.random.default_rng(42)

        # Initial population of 10 points spread across the domain
        pop_size = 10
        population = []
        for i in range(pop_size):
            x = rng.uniform(-5, 5, dim)
            fx = safe_eval(x)
            if fx is None:
                break
            population.append((fx, x))

        if not population:
            return best_x, best_f

        # Sort by fitness
        population.sort(key=lambda p: p[0])
        best_x, best_f = population[0][1], population[0][0]

        # Adaptive sigma
        sigma = 1.5

        # Main loop: population-based evolution
        while call_count < budget and time_ok():
            # Generate offspring around the top few individuals
            n_offspring = min(pop_size, budget - call_count)
            if n_offspring <= 0:
                break

            offspring = []
            for i in range(n_offspring):
                # Pick a parent: biased toward better individuals
                if rng.random() < 0.7:
                    parent = population[0][1]
                elif population:
                    parent = population[rng.integers(0, min(3, len(population)))][1]
                else:
                    parent = best_x

                # Adaptive step size
                y = parent + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_eval(y)
                if fy is not None:
                    offspring.append((fy, y))

            if not offspring:
                break

            # Combine population and offspring, keep best pop_size
            combined = population + offspring
            combined.sort(key=lambda p: p[0])
            population = combined[:pop_size]

            # Update best
            if population[0][0] < best_f:
                best_f = population[0][0]
                best_x = population[0][1].copy()

            # Adapt sigma based on improvement
            old_best = population[0][0]
            # Check if we made progress
            progress = any(o[0] < population[0][0] for o in offspring)
            if progress:
                sigma *= 1.2
            else:
                sigma *= 0.85
            sigma = max(0.01, min(sigma, 10.0))

        # Phase 2: Local refinement with Nelder-Mead around the best point
        if best_x is not None and time_ok() and call_count < budget:
            best_x = self._nelder_mead(safe_eval, best_x.copy(), budget, dim, deadline)

        # Phase 3: Final fine-tuning with small perturbations
        if best_x is not None and time_ok() and call_count < budget:
            fine_sigma = 0.1
            for _ in range(min(50, budget - call_count)):
                if not time_ok():
                    break
                y = best_x + fine_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_eval(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                else:
                    fine_sigma *= 0.9
                    fine_sigma = max(fine_sigma, 0.001)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f

    def _nelder_mead(self, safe_eval, x0, budget, dim, deadline):
        """Nelder-Mead simplex method for local refinement."""
        # Initialize simplex
        simplex = [x0.copy()]
        f0 = safe_eval(x0)
        if f0 is None:
            return x0

        simplex.append((f0, x0.copy()))

        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5
            fi = safe_eval(xi)
            if fi is None:
                break
            simplex.append((fi, xi.copy()))

        # Sort simplex
        simplex.sort(key=lambda p: p[0])

        # Main NM loop
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        max_iter = min(200, budget)
        for _ in range(max_iter):
            if time.time() >= deadline:
                break

            # Sort
            simplex.sort(key=lambda p: p[0])
            x_worst = simplex[-1][1]
            x_best = simplex[0][1]
            f_worst = simplex[-1][0]
            f_best = simplex[0][0]

            # Check convergence
            if abs(f_worst - f_best) < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean([p[1] for p in simplex[:-1]], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = safe_eval(x_r)
            if f_r is None:
                break

            if f_r < f_best:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = safe_eval(x_e)
                if f_e is None:
                    break
                if f_e < f_r:
                    simplex[-1] = (f_e, x_e)
                else:
                    simplex[-1] = (f_r, x_r)
            elif f_r < f_worst:
                simplex[-1] = (f_r, x_r)
            else:
                # Contraction
                x_c = centroid + rho * (x_worst - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = safe_eval(x_c)
                if f_c is None:
                    break
                if f_c < f_worst:
                    simplex[-1] = (f_c, x_c)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i][1] = x_best + sigma_nm * (simplex[i][1] - x_best)
                        simplex[i][1] = np.clip(simplex[i][1], -5.0, 5.0)
                        fi = safe_eval(simplex[i][1])
                        if fi is None:
                            return simplex[0][1]
                        simplex[i][0] = fi

        # Return best
        simplex.sort(key=lambda p: p[0])
        return simplex[0][1]
# EVOLVE-BLOCK-END
