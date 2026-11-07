"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Multi-restart CMA-ES with (1+1)-ES refinement for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to find a good starting point
        n_init = min(30, budget // 10)
        for i in range(n_init):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if global_best_x is None:
            global_best_x = np.zeros(dim)
            evaluate(global_best_x)

        # Phase 2: CMA-ES with restarts, followed by (1+1)-ES refinement
        remaining_budget = budget - calls_made

        n_restarts = 4
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if time.time() > deadline or calls_made >= budget:
                break
            if budget_per_restart < 20:
                break

            # Initialize CMA-ES parameters
            if restart == 0:
                x_mean = global_best_x.copy()
            else:
                # For subsequent restarts, start from a random point or slightly perturb global best
                if rng.random() < 0.5:
                    x_mean = global_best_x + rng.normal(0, 1.0, dim)
                    x_mean = np.clip(x_mean, -5, 5)
                else:
                    x_mean = rng.uniform(-5, 5, dim)
                fx = evaluate(x_mean)

            # CMA-ES parameters
            sigma = 1.5  # initial step size
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            C = np.eye(dim)  # covariance matrix
            p_sigma = np.zeros(dim)  # evolution path for sigma
            p_c = np.zeros(dim)  # evolution path for C

            # Adaptation parameters
            mu = lam // 2  # number of best individuals
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Local best for this restart
            local_best_x = x_mean.copy()
            local_best_f = global_best_f if restart == 0 else fx

            calls_in_restart = 0
            max_calls = budget_per_restart

            # Precompute Cholesky factor for efficiency
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim)

            for gen in range(max_calls // lam + 1):
                if time.time() > deadline or calls_made >= budget:
                    break
                if calls_in_restart >= max_calls:
                    break

                # Generate population
                xs = np.zeros((lam, dim))
                fs = np.zeros(lam)

                for i in range(lam):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    xs[i] = x
                    fx = evaluate(x)
                    fs[i] = fx
                    calls_in_restart += 1

                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()

                if calls_in_restart >= max_calls:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                xs_sorted = xs[idx]
                fs_sorted = fs[idx]

                # Update mean
                x_mean_new = np.sum(w[:, np.newaxis] * xs_sorted[:mu], axis=0)

                # Update sigma
                y = (x_mean_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(dim)))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                delta_h = (1 - c_c) ** 2 - c_c * (2 - c_c)
                C = (1 + c_1 * (mu_eff + 2) - c_1) * C
                C += c_1 * (np.outer(y, y) - delta_h * C)
                C += c_mu * np.sum(w[:mu, np.newaxis] * np.outer(xs_sorted[:mu] - x_mean, xs_sorted[:mu] - x_mean), axis=0) / sigma ** 2

                # Ensure positive semi-definite
                C = 0.5 * (C + C.T)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                except:
                    C = np.eye(dim)
                    L = np.eye(dim)

                x_mean = x_mean_new

            # After CMA-ES restart, if local best improved global, update
            if local_best_f < global_best_f:
                global_best_f = local_best_f
                global_best_x = local_best_x.copy()

            # Phase 3: (1+1)-ES refinement on the best point found so far
            if global_best_x is not None and calls_made < budget and time.time() < deadline:
                x = global_best_x.copy()
                fx = global_best_f
                sigma_es = 0.5
                window = 10
                successes = 0
                total = 0
                no_improve_count = 0
                max_no_improve = 30

                while calls_made < budget and time.time() < deadline:
                    y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)

                    if fy < fx:
                        x, fx = y, fy
                        successes += 1
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    total += 1
                    if total >= window:
                        rate = successes / total
                        if rate > 0.2:
                            sigma_es *= 1.3
                        elif rate < 0.15:
                            sigma_es *= 0.7
                        successes = 0
                        total = 0

                    sigma_es = np.clip(sigma_es, 0.01, 5.0)

                    if no_improve_count >= max_no_improve:
                        break

        # Final polish: use scipy L-BFGS-B on the best point found so far
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                polish_budget = budget - calls_made
                if polish_budget > 10:
                    def polished_func(z):
                        nonlocal calls_made
                        if calls_made >= budget:
                            return func(z)
                        calls_made += 1
                        return func(z)

                    res = scipy_minimize(polished_func, global_best_x, method='L-BFGS-B',
                                        bounds=[(-5, 5)] * dim,
                                        options={'maxiter': min(polish_budget, 50),
                                                 'maxfun': min(polish_budget, 50),
                                                 'gtol': 1e-6})
                    if res.fun < global_best_f:
                        global_best_f = res.fun
                        global_best_x = res.x.copy()
            except Exception:
                pass

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
