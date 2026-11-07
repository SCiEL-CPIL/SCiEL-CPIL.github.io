"""Multi-restart CMA-ES for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline - 2.0 and calls_used < budget

        def safe_call(x):
            nonlocal calls_used
            if not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return None
            calls_used += 1
            return x.copy(), fx

        def cma_es(x0, sigma0, max_evals, seed):
            """Run CMA-ES from x0 with initial sigma0, using up to max_evals evaluations."""
            nonlocal calls_used, best_x, best_fx

            rng_local = np.random.default_rng(seed)

            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
            cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

            x = np.array(x0, dtype=np.float64)
            sigma = sigma0

            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)

            # Cholesky of C
            try:
                csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                csqrt = np.eye(dim)

            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            local_best_x = x.copy()
            local_best_fx = np.inf
            local_calls = 0
            no_improve = 0

            # Initial evaluation of x0 if not already done
            # We'll just start generating offspring

            while local_calls < max_evals and time_ok():
                # Generate population
                pop = []
                for i in range(lam):
                    if not time_ok() or local_calls >= max_evals:
                        break
                    z = rng_local.standard_normal(dim)
                    xi = x + sigma * (csqrt @ z)
                    result = safe_call(xi)
                    if result is None:
                        break
                    xi_c, fi = result
                    pop.append((fi, xi_c))

                if len(pop) < mu:
                    break

                pop.sort(key=lambda t: t[0])

                # Update local best
                if pop[0][0] < local_best_fx:
                    local_best_fx = pop[0][0]
                    local_best_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update global best
                if pop[0][0] < best_fx:
                    best_fx = pop[0][0]
                    best_x = pop[0][1].copy()

                # Mean update
                x_new = np.sum([weights[i] * pop[i][1] for i in range(mu)], axis=0)

                # Path updates
                y = (x_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (local_calls + 1)))) < \
                       (1 + 2 / (dim + 1)) * chi_n

                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * y

                # Covariance update
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    yi = (pop[i][1] - x) / sigma
                    rankmu += weights[i] * np.outer(yi, yi)

                C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Ensure positive definiteness
                try:
                    csqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    # Fallback: eigenvalue decomposition
                    w, V = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-14)
                    C = V @ np.diag(w) @ V.T
                    try:
                        csqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    except np.linalg.LinAlgError:
                        csqrt = np.eye(dim)

                # Sigma update
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                x = x_new
                local_calls += 1

                # Restart if stagnated
                if no_improve > 15 and local_calls > lam * 3:
                    break

            return local_best_x, local_best_fx, local_calls

        # Phase 1: Initial exploration with a few random points
        n_init = min(10, budget)
        for _ in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-4, 4, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            if fx < best_fx:
                best_fx = fx
                best_x = x_c.copy()

        # Phase 2: Multi-restart CMA-ES
        # First restart from the best point found
        remaining = budget - calls_used
        if remaining > 10:
            # Use a moderate initial sigma
            sigma0 = 1.0
            max_evals_first = min(remaining, 200)
            _, _, used = cma_es(best_x, sigma0, max_evals_first, seed=123)
            remaining = budget - calls_used

            # If we still have budget, do more restarts from different points
            if remaining > 20:
                # Generate several diverse starting points
                n_restarts = min(5, remaining // 20)
                for r in range(n_restarts):
                    if not time_ok():
                        break
                    # Start from a point near the current best but perturbed
                    x_start = best_x + rng.normal(0, 1.5, dim)
                    x_start = np.clip(x_start, -5, 5)
                    sigma0 = 0.5
                    max_evals_r = min(remaining - r * 10, 80)
                    if max_evals_r < 10:
                        break
                    _, _, _ = cma_es(x_start, sigma0, max_evals_r, seed=1000 + r)
                    remaining = budget - calls_used

        # Phase 3: Final local refinement using a small CMA-ES with very small sigma
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            sigma0 = 0.05
            max_evals_final = min(remaining, 50)
            _, _, _ = cma_es(best_x, sigma0, max_evals_final, seed=9999)

        return best_x, best_fx
# EVOLVE-BLOCK-END
