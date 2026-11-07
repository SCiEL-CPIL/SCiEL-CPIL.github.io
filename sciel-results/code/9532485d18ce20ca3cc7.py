"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimizer with CMA-ES and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 35s safety margin before 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), -5, 5)
            if len(x) != dim:
                x = x[:dim]
            try:
                val = func(x)
            except Exception:
                return np.inf
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, maxfev=15):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            bounds = [(-5.0, 5.0) for _ in range(dim)]
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': maxfev,
                        'ftol': 1e-16,
                        'gtol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        def run_cmaes(x0, maxevals, sigma0=1.0):
            """Run a simple CMA-ES implementation."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None

            n = dim
            lambda_pop = max(4 + int(3 * np.log(n)), 10)
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            local_best_x = x0.copy()
            local_best_f = wrapped_func(x0)

            eval_count = 0
            stagnation = 0
            last_best = local_best_f

            while eval_count < maxevals and calls < budget and time.time() < deadline:
                # Generate population
                pop = []
                try:
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    x_new = x_mean + sigma * (B @ z)
                    x_new = np.clip(x_new, -5, 5)
                    pop.append(x_new)

                # Evaluate population
                pop_f = []
                for x in pop:
                    if calls >= budget or time.time() > deadline:
                        break
                    f = wrapped_func(x)
                    pop_f.append(f)
                    eval_count += 1
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()

                if len(pop_f) < lambda_pop:
                    break

                # Sort by fitness
                indices = np.argsort(pop_f)
                pop = [pop[i] for i in indices]
                pop_f = [pop_f[i] for i in indices]

                # Update mean
                x_mean_new = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)

                # Update step size control
                try:
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    D = np.diag(B)
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / D @ (B.T @ (x_mean_new - x_mean) / sigma)

                    if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                    # Update covariance
                    rank_mu = np.outer(x_mean_new - x_mean, x_mean_new - x_mean)
                    rank_one = np.outer(x_mean - x_mean_new, x_mean - x_mean_new)

                    C = (1 + c_1 + c_mu - c_c) * C + c_1 * rank_one + c_mu * rank_mu
                    C = (C + C.T) / 2
                except np.linalg.LinAlgError:
                    sigma *= 0.9

                x_mean = x_mean_new
                x_mean = np.clip(x_mean, -5, 5)

                # Check for stagnation
                if abs(local_best_f - last_best) < 1e-12:
                    stagnation += 1
                else:
                    stagnation = 0
                    last_best = local_best_f

                if stagnation > 20:
                    break

            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Fast initial sampling - fewer points, start CMA-ES sooner
        n_initial = 8
        init_points = [np.zeros(dim)]
        # Add a few structured points
        for axis in range(min(dim, 2)):
            pt = np.zeros(dim)
            pt[axis] = 1.5
            init_points.append(pt)
        # Fewer random points
        for i in range(n_initial - len(init_points)):
            init_points.append(rng.uniform(-3, 3, dim))

        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Quick polish from initial best
        if budget - calls > 10:
            polish(best_x, maxfev=min(10, (budget - calls) // 15))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Main CMA-ES run with generous budget
        remaining = budget - calls
        cma_budget = int(remaining * 0.7)
        if cma_budget >= 20:
            run_cmaes(best_x.copy(), cma_budget, sigma0=0.7)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Polish from CMA-ES result
        if budget - calls > 10:
            polish(best_x.copy(), maxfev=min(20, budget - calls))

        # Phase 4: Conditional restarts only if stuck
        if budget - calls > 25:
            # Check if we're stuck (best hasn't improved much recently)
            n_restarts = min(3, (budget - calls) // 30)
            sigma_values = [0.3, 0.5, 0.8]
            for r in range(n_restarts):
                if calls >= budget or time.time() > deadline:
                    break
                x_start = best_x + rng.standard_normal(dim) * 0.3
                x_start = np.clip(x_start, -5, 5)
                sigma_r = sigma_values[r % len(sigma_values)]
                run_cmaes(x_start, min(30, budget - calls), sigma0=sigma_r)

        # Final high-precision polish
        if budget - calls > 5:
            polish(best_x.copy(), maxfev=min(15, budget - calls))

        return best_x, best_f
# EVOLVE-BLOCK-END
