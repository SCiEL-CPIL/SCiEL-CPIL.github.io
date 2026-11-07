"""Optimizer for BBOB f14 (Different powers) using CMA-ES with adaptive restarts."""
import numpy as np
import time


class Optimizer:
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with
    adaptive restarts for robust optimization on shifted/rotated instances."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cmaes(x0, sigma0, max_evals, seed_offset=0):
            """Run CMA-ES from x0 with initial step size sigma0."""
            nonlocal call_count, best_x, best_f, deadline
            rng = np.random.default_rng(42 + seed_offset)

            dim = n
            x = x0.astype(float).copy()
            sigma = sigma0

            # CMA-ES parameters
            lambda_pop = 4 + int(3 * np.log(dim))  # population size
            mu = lambda_pop // 2  # number of recombination weights
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            # State variables
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)
            # Cholesky of C
            try:
                D = np.diag(np.sqrt(np.diag(C)))
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)

            # Initial evaluations
            evals_used = 0
            pop_xs = []
            pop_fs = []

            for gen in range(1, max_evals // lambda_pop + 100):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate population
                pop_xs = []
                pop_fs = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    xi = x + sigma * (B @ z)
                    fx = eval_func(xi)
                    if fx == np.inf:
                        pop_xs.append(xi)
                        pop_fs.append(np.inf)
                    else:
                        pop_xs.append(xi)
                        pop_fs.append(fx)
                    evals_used += 1
                    if call_count >= budget or time.time() > deadline:
                        break

                if len(pop_xs) < 2:
                    break

                # Sort by fitness
                order = np.argsort(pop_fs)
                pop_xs = [pop_xs[i] for i in order]
                pop_fs = [pop_fs[i] for i in order]

                # Weighted mean
                x_new = np.sum([weights[i] * pop_xs[i] for i in range(mu)], axis=0)

                # Update step size
                if evals_used > 0:
                    # Update ps
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (x_new - x))
                    hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used / lambda_pop))) < \
                           (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                    if hsig:
                        pc = (1 - cs) * pc + cs * (x_new - x) / sigma
                    sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                    sigma = np.clip(sigma, 1e-10, 1e10)

                    # Update covariance
                    delta_sig = (1 - hsig) * cs * (2 - cs)
                    pc_update = np.outer(pc, pc)
                    C = (1 + c1 + cmu - c1 - cmu) * C + \
                        c1 * (np.outer(x_new - x, x_new - x) + (1 - hsig) * delta_sig * np.outer(pc, pc) - pc) + \
                        cmu * np.sum([weights[i] * np.outer(pop_xs[i] - x, pop_xs[i] - x) for i in range(mu)], axis=0)

                    # Ensure symmetry
                    C = (C + C.T) / 2

                    # Recompute B and D
                    try:
                        B = np.linalg.cholesky(C + 1e-15 * np.eye(dim))
                        D = np.diag(B)
                    except np.linalg.LinAlgError:
                        B = np.eye(dim)
                        D = np.ones(dim)

                x = x_new

                # Check convergence
                if np.std(pop_fs[:mu]) < 1e-10:
                    break
                if sigma < 1e-10:
                    break

            return x, min(pop_fs) if pop_fs else np.inf

        # Initial diverse sampling
        rng = np.random.default_rng(42)
        n_init = 30
        points = []
        for i in range(n_init):
            x = rng.uniform(-4.5, 4.5, n)
            fx = eval_func(x)
            points.append((x, fx))

        points.sort(key=lambda t: t[1])
        x_best_init = points[0][0].copy()
        f_best_init = points[0][1]

        # Run CMA-ES from best initial point
        x_cur, f_cur = cmaes(x_best_init, 1.0, max_evals=budget - call_count, seed_offset=0)

        # If we have budget left, try a few more restarts
        remaining = budget - call_count
        if remaining > 50:
            for restart in range(3):
                if call_count >= budget or time.time() > deadline:
                    break
                # New random start near current best
                x_start = x_cur + rng.normal(0, 0.3, n)
                x_start = np.clip(x_start, -5, 5)
                f_start = eval_func(x_start)
                if f_start < f_cur:
                    x_cur, f_cur = cmaes(x_start, 0.5, max_evals=remaining - call_count, seed_offset=100 + restart)
                remaining = budget - call_count
                if remaining < 30:
                    break

        return best_x, best_f
