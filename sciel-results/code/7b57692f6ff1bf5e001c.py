"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-Based CMA-ES with multi-restart and local refinement for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def cma_es_population(x0, sigma0, max_calls):
            """
            Population-based CMA-ES with covariance adaptation.
            Maintains a population of lambda candidates per generation.
            """
            nonlocal calls_made, best_x, best_fx

            d = self.dim
            x = x0.copy().astype(float)
            sigma = sigma0

            # CMA-ES parameters
            lambda_pop = max(4, int(4 + 3 * np.log(d)))
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (d + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

            # Covariance matrix
            C = np.eye(d)
            pc = np.zeros(d)
            cs = np.zeros(d)

            # Initial evaluation
            fx = float(func(x))
            calls_made += 1
            update_best(x, fx)

            # Decompose C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            best_local = fx
            stagnation = 0
            gen = 0

            while calls_made < calls_made + max_calls and not check_time():
                gen += 1
                if calls_made >= self.budget:
                    break

                # Generate population
                candidates = []
                for i in range(lambda_pop):
                    if check_time() or calls_made >= self.budget:
                        break
                    z = rng.standard_normal(d)
                    y = x + sigma * (B @ (D * z))
                    y_clipped = np.clip(y, -5.0, 5.0)
                    candidates.append(y_clipped)

                if len(candidates) < lambda_pop:
                    break

                # Evaluate population
                fvals = []
                for y in candidates:
                    if check_time() or calls_made >= self.budget:
                        break
                    fy = float(func(y))
                    calls_made += 1
                    fvals.append(fy)
                    if fy < best_local:
                        best_local = fy

                if len(fvals) < lambda_pop:
                    break

                # Sort by fitness
                order = np.argsort(fvals)

                # Weighted mean of best mu
                x_w = np.zeros(d)
                for i in range(mu):
                    x_w += weights[i] * candidates[order[i]]

                # Update step size (CSA)
                y = (x_w - x) / sigma
                cs = (1 - c_sigma) * cs + c_sigma * (np.linalg.norm(y) / np.sqrt(2 * d) - 1)
                sigma *= np.exp(c_sigma * d_sigma * cs)
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update evolution paths
                pc = (1 - c_c) * pc + c_c * y
                h_sigma = (1 - (1 - c_sigma) ** gen) / (1 - (1 - c_c) ** gen)
                threshold = (1.4 + 1 / d) * np.sqrt(d)
                if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** gen) < threshold:
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (1 - c_c)) * C)
                    for i in range(mu):
                        zi = (candidates[order[i]] - x) / sigma
                        C += c_mu * weights[i] * np.outer(zi, zi)

                # Recalculate decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                x = x_w
                fx = min(fvals)
                update_best(x, fx)

                # Stagnation check
                if fx < best_local - 1e-10:
                    best_local = fx
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 15:
                    break

            return x, fx

        # Phase 1: Multi-restart with population-based CMA-ES
        num_restarts = 5
        calls_per_restart = (self.budget - 30) // num_restarts

        for r in range(num_restarts):
            if check_time() or calls_made >= self.budget:
                break

            remaining = self.budget - calls_made
            if remaining < 10:
                break

            # Initialize: first restart random, subsequent near best with perturbation
            if best_x is None:
                x0 = rng.uniform(-5.0, 5.0, self.dim)
                sigma0 = 2.0
            else:
                spread = 1.0 + r * 0.3
                x0 = np.clip(best_x + spread * rng.standard_normal(self.dim), -5.0, 5.0)
                sigma0 = 0.8 + r * 0.2

            x, fx = cma_es_population(x0, sigma0, calls_per_restart)

        # Phase 2: Local refinement with coordinate search
        if best_x is not None and calls_made < self.budget and not check_time():
            x = best_x.copy()
            fx = best_fx

            # Coordinate-wise local search with adaptive step
            for coord in range(self.dim):
                if check_time() or calls_made >= self.budget:
                    break
                for direction in [1, -1]:
                    if check_time() or calls_made >= self.budget:
                        break
                    step = 0.3
                    for _ in range(20):
                        if check_time() or calls_made >= self.budget:
                            break
                        y = x.copy()
                        y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                        fy = float(func(y))
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)
                            step *= 0.6
                        else:
                            break

            # Fine-tuning with small random steps
            sigma = 0.05
            for i in range(min(50, self.budget - calls_made)):
                if check_time() or calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = float(func(y))
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(self.dim)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
