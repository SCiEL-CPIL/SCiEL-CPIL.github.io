"""Optimizer for BBOB-style f11 (Discus) with CMA-ES + coordinate descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES with adaptive restarts + coordinate descent polish for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 330s kill

        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            call_count += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if not np.isfinite(f):
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(12345)

        # Phase 1: Initial random sampling to find a good starting region
        n_random = min(30, budget)
        for _ in range(n_random):
            if time.time() > deadline or call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES optimization
        def cma_es(x0, max_evals):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline or call_count >= budget:
                return

            x = x0.copy()
            # Initial sigma based on spread of initial samples
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)

            # CMA-ES parameters
            mu = max(1, dim // 2)  # number of selected individuals
            lambda_ = 2 * dim  # population size
            weights = np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)

            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            chiN = math.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

            # Track stagnation
            no_improve = 0
            last_best = best_f
            stop_threshold = 1e-10

            for gen in range(max_evals // lambda_ + 1):
                if time.time() > deadline or call_count >= budget:
                    break

                # Check for stagnation
                if best_f < last_best - 1e-8:
                    no_improve = 0
                    last_best = best_f
                else:
                    no_improve += 1

                if no_improve > 20 and sigma < 1e-6:
                    break

                # Generate population
                pop = []
                fpop = []
                for i in range(lambda_):
                    if time.time() > deadline or call_count >= budget:
                        break
                    z = rng.standard_normal(dim)
                    xi = x + sigma * (C @ z)
                    xi = np.clip(xi, -5, 5)
                    fi = safe_call(xi)
                    pop.append(xi)
                    fpop.append(fi)

                if len(pop) < mu:
                    break

                # Sort by fitness
                order = np.argsort(fpop)
                pop = [pop[i] for i in order]
                fpop = [fpop[i] for i in order]

                # Update mean
                x_old = x.copy()
                x_new = sum(weights[i] * pop[i] for i in range(mu))

                # Update covariance matrix
                # pc update
                pc = (1 - cc) * pc + cc * (x_new - x_old) / sigma

                # cs update
                cs = (1 - cs) * cs + cs * (np.linalg.norm(x_new - x_old) / chiN - 1)

                # Update C
                delta = 1 + 2 * max(0, (cs - 1) / (1 - cs)) - 1
                C = ((1 - c1 - cmu + delta * c1) * C +
                     c1 * (np.outer(pc, pc) - C + (cc * (2 - cc)) * np.eye(dim)) +
                     cmu * sum(weights[i] * np.outer(pop[i] - x_old, pop[i] - x_old)
                               for i in range(mu)))

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= math.exp(cs * (np.linalg.norm(x_new - x_old) / chiN - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                x = x_new

                # Check convergence
                if sigma < 1e-8:
                    break

                # Stagnation detection
                if np.max(fpop) - np.min(fpop) < 1e-12 and gen > 5:
                    break

        # Run CMA-ES from the best initial point
        cma_es(best_x, budget - call_count)

        # Phase 3: Coordinate descent polish
        if best_x is not None and time.time() < deadline and call_count < budget:
            x = best_x.copy()
            fx = safe_call(x)
            step = 0.5
            for _ in range(150):
                if time.time() > deadline or call_count >= budget:
                    break
                improved = False
                for i in range(dim):
                    if time.time() > deadline or call_count >= budget:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_call(x_try)
                        if f_try < fx - 1e-12:
                            x = x_try
                            fx = f_try
                            improved = True
                            break
                if not improved:
                    step *= 0.7
                    if step < 1e-5:
                        break
                else:
                    step = max(step * 0.9, 1e-5)

        return best_x, best_f
# EVOLVE-BLOCK-END
