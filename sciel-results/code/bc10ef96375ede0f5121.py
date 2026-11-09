"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with population-based search and restart strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def record(x, f):
            nonlocal best_x, best_f, calls_used
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Initial exploration with diverse points ---
        n_init = min(20, budget // 10)
        init_points = np.vstack([
            np.zeros(dim),
            rng.uniform(-5, 5, size=(n_init - 1, dim))
        ])
        for i in range(n_init):
            if calls_used >= budget or time.time() > deadline:
                break
            x = init_points[i]
            f = float(func(x))
            record(x, f)

        # --- Phase 2: CMA-ES ---
        def cma_es(x_start, sigma_start, budget_left):
            nonlocal calls_used, best_x, best_f
            if budget_left <= 0 or time.time() > deadline:
                return best_x, best_f

            n = dim
            # Population size: lambda = 4 + floor(3 * ln(n))
            lam = 4 + int(3 * np.log(max(n, 2)))
            mu = lam // 2  # Number of parents

            # CMA-ES parameters
            c_sigma = (n + 2) / (n + 5.5 + mu)
            c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * ((mu + 2) / (n + 2)) ** 2 - c_1)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1) - 1)) + c_sigma

            # State variables
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            gen = 0
            while calls_used < budget and time.time() < deadline:
                # Generate population
                B = np.linalg.cholesky(C)
                z = rng.standard_normal((lam, n))
                y = z @ B
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                offspring = x_mean + sigma * y

                # Evaluate offspring
                offspring_f = []
                for i in range(lam):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    xi = np.clip(offspring[i], -5, 5)
                    fi = float(func(xi))
                    record(xi, fi)
                    offspring_f.append(fi)

                if len(offspring_f) < lam:
                    break

                offspring_f = np.array(offspring_f)
                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring = offspring[idx]
                offspring_f = offspring_f[idx]

                # Update mean
                weights = np.zeros(mu)
                weights[0] = np.log(mu) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                w_eff = 1 / np.sum(weights ** 2)

                y_w = np.sum(weights[:, None] * offspring[:mu], axis=0)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * w_eff) / np.linalg.norm(B) * y_w
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) / chi_n) < 1.4 + 2 / (n + 1)
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * w_eff) * y_w

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu + delta_h) * C
                C += c_1 * np.outer(pc, pc)
                C += c_mu * np.sum(weights[:, None, None] * np.outer(offspring[:mu] - x_mean, offspring[:mu] - x_mean), axis=0)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update mean
                x_mean = x_mean + sigma * y_w
                x_mean = np.clip(x_mean, -5, 5)

                gen += 1

                # Check for stagnation: if best hasn't improved in last 2*lam generations, restart
                if gen % (2 * lam) == 0:
                    if best_f > 1e-6 and np.linalg.norm(x_mean - best_x) < sigma * 0.1:
                        break

            return best_x, best_f

        # Run CMA-ES with the best initial point
        x_start = best_x.copy()
        # Estimate initial sigma from initial spread
        if n_init > 1:
            spreads = []
            for d in range(dim):
                vals = [init_points[i, d] for i in range(n_init)]
                spreads.append(np.std(vals))
            sigma_start = max(0.5, min(2.0, np.mean(spreads) * 0.5))
        else:
            sigma_start = 1.0

        cma_es(x_start, sigma_start, budget - calls_used)

        # --- Phase 3: Fine-tuning with (1+1)-ES if budget remains ---
        if calls_used < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.01, sigma_start * 0.1)
            successes = []
            window_size = 15

            while calls_used < budget and time.time() < deadline:
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = float(func(y))
                calls_used += 1

                success = fy < fx
                successes.append(success)
                if len(successes) > window_size:
                    successes.pop(0)

                if success:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                if len(successes) >= 5:
                    rate = np.mean(successes)
                    if rate > 0.2:
                        sigma *= 1.15
                    elif rate < 0.2:
                        sigma *= 0.85
                    sigma = np.clip(sigma, 0.001, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
