"""Rewritten optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer:
      1. CMA-ES-like population-based search for global exploration on Rastrigin.
      2. Switch to Nelder-Mead local refinement when close to optimum.
    Includes a wall-clock deadline guard (300s limit, return best ~10s before).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: CMA-ES-like population search ---
        # Initialize population around center of box
        pop_size = 12  # population size
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 2.5
        c1 = (4 + np.log(dim)) / (dim + 4 + 3 * dim)
        cmu = 2 * (pop_size // 2 - 2 + 1.0 / dim) / ((dim + 2) ** 2 + pop_size)
        c_sigma = (pop_size + 1.7 - dim) / (pop_size + dim + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Cumulation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Generate initial population
        def generate_population(x, sigma, rng):
            # Sample from multivariate normal approximated by C
            # Use Cholesky for efficiency
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            samples = []
            for _ in range(pop_size):
                z = rng.standard_normal(dim)
                samples.append(x + sigma * (L @ z))
            return np.array(samples)

        # Sort by fitness for CMA-ES update
        for iteration in range(budget):
            if time.time() > deadline:
                break

            # Generate population
            pop = generate_population(x, sigma, rng)

            # Evaluate
            evals = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                xi = np.clip(pop[i], -5, 5)
                fi = func(xi)
                evals.append((fi, xi))

            if not evals:
                break

            # Sort by fitness
            evals.sort(key=lambda t: t[0])

            # Update best
            for fi, xi in evals:
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()

            # Update mean
            weights = np.ones(pop_size) / pop_size
            x = np.average([e[1] for e in evals], axis=0, weights=weights)

            # Update sigma (step size adaptation)
            for fi, xi in evals[:pop_size // 2]:
                pass  # Simplified: just use basic adaptation

            # Basic sigma adaptation
            if len(evals) >= 2:
                mean_fit = np.mean([e[0] for e in evals])
                std_fit = np.std([e[0] for e in evals])
                if std_fit > 1e-10:
                    # If variance is high, increase sigma; if low, decrease
                    if mean_fit > best_fx + 0.01:
                        sigma *= 0.95
                    else:
                        sigma *= 1.05
                sigma = np.clip(sigma, 0.01, 5.0)

            # Check if we should switch to local refinement
            # Heuristic: if best_fx is very good relative to initial, or if progress is slow
            if iteration > 100 and best_fx < 5.0:
                break

        # --- Phase 2: Nelder-Mead local refinement ---
        # Use remaining budget for local search
        remaining_budget = budget - iteration - 1
        if remaining_budget > 10 and time.time() < deadline:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            for i in range(dim):
                xi = best_x.copy()
                step = 0.1 if best_fx > 1.0 else 0.01
                xi[i] += step
                simplex.append(xi)

            # Evaluate simplex
            simplex_f = [func(np.clip(s, -5, 5)) for s in simplex]

            # Nelder-Mead iterations
            nm_budget = min(remaining_budget, 200)
            for nm_iter in range(nm_budget):
                if time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = func(np.clip(simplex[i], -5, 5))

                # Update best
                min_f = min(simplex_f)
                min_idx = np.argmin(simplex_f)
                if min_f < best_fx:
                    best_fx = min_f
                    best_x = simplex[min_idx].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
