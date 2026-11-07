"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 320.0  # Safety margin before 330s
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State variables
        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, f_best, x_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # Phase 1: Initial random sampling to find a good starting point
        n_init = min(20, budget // 5)
        if n_init > 0:
            for i in range(n_init):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                evaluate(x)

        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        # Phase 2: CMA-ES with restarts
        lambda_pop = max(10, 4 + int(3 * dim))  # Population size
        sigma = 1.5  # Initial step size
        c_sigma = (lambda_pop + 2) / (dim + lambda_pop + 5)
        c_c = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * ((lambda_pop / 2 - 1 + 1.0 / dim) / (dim + 2) ** 2 + lambda_pop))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * ((lambda_pop / 2 - 1 + 1.0 / dim) / (dim + 2) ** 2 + lambda_pop))

        def cma_es_run(x0, sigma0, max_iters):
            nonlocal calls_made, f_best, x_best
            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            it = 0
            no_improve = 0
            best_in_run = f_best

            for it in range(max_iters):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_pop):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    # Whitening transform
                    y = np.linalg.cholesky(C).T @ z
                    xi = x + sigma * y
                    fi = evaluate(xi)
                    offspring.append((fi, xi))

                if not offspring:
                    break

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_in_run:
                    best_in_run = offspring[0][0]
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                w = weights[:len(offspring)]
                if len(offspring) < mu:
                    break
                x_new = sum(w[i] * offspring[i][1] for i in range(mu))

                # Update step size
                d_sigma = 1.0
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C ** -0.5) @ (x_new - x)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update covariance
                pc = (1 - c_c) * pc + c_c * (x_new - x)
                delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < (1 + 2 / dim) * chi_n else 0
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * sum(
                    weights[i] * np.outer(offspring[i][1] - x, offspring[i][1] - x) for i in range(mu)
                )

                # Symmetrize
                C = (C + C.T) / 2
                x = x_new

                # Restart if stagnation
                if no_improve > 30 or sigma < 1e-8:
                    break

            return x, best_in_run

        # Main CMA-ES loop with restarts
        max_restarts = 5
        for restart in range(max_restarts):
            remaining = budget - calls_made
            if remaining < 20 or time.time() > deadline:
                break

            # Determine iteration budget for this run
            iters = min(remaining, max(30, remaining // 2))

            # Start from current best or a new random point for later restarts
            if restart == 0:
                x0 = x_best.copy()
                sigma0 = 1.0
            else:
                # Explore around best with some perturbation
                x0 = x_best + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.5 * (1 + 0.3 * restart)

            x_new, f_new = cma_es_run(x0, sigma0, iters)

            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new

            if time.time() > deadline or calls_made >= budget:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
