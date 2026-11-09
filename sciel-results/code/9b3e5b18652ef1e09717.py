"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + (1+1)-ES hybrid optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 325.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, n_evals, max_gen=100):
            """Run CMA-ES for up to n_evals function calls."""
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return

            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of selected parents
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            evals_in_run = 0

            while evals_in_run < n_evals and calls < budget and time.time() < deadline:
                # Generate population
                Bs = np.linalg.cholesky(C)
                y = np.array([
                    rng.standard_normal(n)
                    for _ in range(lam)
                ])
                z = np.array([Bs @ y[i] for i in range(lam)])
                x_pop = np.array([x_mean + sigma * z[i] for i in range(lam)])
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    fx = try_call(x_pop[i])
                    if fx is None:
                        return
                    f_pop.append(fx)
                    evals_in_run += 1
                else:
                    # All evaluated
                    pass

                if len(f_pop) < lam:
                    # Pad with large values for sorting
                    f_pop = f_pop + [np.inf] * (lam - len(f_pop))

                # Sort by fitness
                order = np.argsort(f_pop)
                y_selected = np.array([y[order[i]] for i in range(mu)])
                x_selected = np.array([x_pop[order[i]] for i in range(mu)])
                f_selected = np.array([f_pop[order[i]] for i in range(mu)])

                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * x_selected, axis=0)

                # Success rule for sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (Bs @ y_selected.T).T / sigma
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update evolution paths
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_old) / sigma
                delta = 0
                if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (n + 1)) * chi_n:
                    delta = 1
                C = (1 - c_1 - c_mu) * C + c_1 * (2 - delta) * np.outer(p_c, p_c) + \
                    c_mu * np.sum(weights[:, None, None] * np.outer(y_selected, y_selected), axis=0)

                # Symmetrize C
                C = (C + C.T) / 2

                # Update best
                if f_selected[0] < best_f:
                    best_f = f_selected[0]
                    best_x = x_selected[0].copy()

                # Check stagnation
                if evals_in_run > 2 * lam and np.std(f_selected[:min(mu, len(f_selected))]) < 1e-10:
                    break

            return x_mean, sigma, best_f

        # Phase 1: Brief random initialization
        n_init = min(5, budget // 10)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            try_call(x)

        # Phase 2: CMA-ES from the best point found
        if best_x is not None:
            cma_budget = min(int(budget * 0.7), budget - calls)
            if cma_budget > 10:
                cma_es(best_x.copy(), 1.5, cma_budget)

        # Phase 3: Final (1+1)-ES for high-precision convergence
        remaining = budget - calls
        if remaining > 5 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            no_improve = 0
            max_no_improve = 30

            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve >= max_no_improve:
                    # Small perturbation to escape local minimum
                    if time.time() > deadline:
                        break
                    x = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                    fx = try_call(x)
                    if fx is None:
                        break
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 0.2
                    no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
