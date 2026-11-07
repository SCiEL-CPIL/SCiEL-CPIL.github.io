"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock) using multi-restart CMA-ES."""
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)

        # Initial exploration: diverse starting points
        init_points = [np.zeros(dim)]
        for _ in range(15):
            init_points.append(rng.uniform(-5, 5, dim))
        for val in [-3, 0, 3]:
            init_points.append(np.full(dim, val))

        for p in init_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 5:
                break
            fx = safe_call(p)
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Multi-restart CMA-ES
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # = 11 for dim=5
        max_restarts = 5
        restart_budget_frac = 0.15  # 15% of remaining budget per restart

        for restart in range(max_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline - 5:
                break

            remaining = budget - call_count
            if remaining < pop_size + 10:
                break

            # For first restart, use best found; for later, sample near best
            if restart == 0:
                x_mean = best_x.copy()
                sigma = 1.0
            else:
                # Sample around best with some spread
                x_mean = best_x + 0.5 * rng.standard_normal(dim)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = 0.8

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs_param = (2 + mueff / dim) / (dim + 2 + mueff / dim)
            c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - cc - cs_param)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            # CMA-ES state
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs_target = 1.0

            # Run CMA-ES until budget or time runs out for this restart
            while call_count + pop_size <= budget:
                if time.time() > deadline - 5:
                    break

                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # C not positive definite, reset
                    C = np.eye(dim)
                    B = np.eye(dim)

                offspring = []
                offspring_f = []

                for i in range(pop_size):
                    if call_count >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) < 2:
                    break

                offspring_f = np.array(offspring_f)
                order = np.argsort(offspring_f)
                offspring = np.array(offspring)[order]
                offspring_f = offspring_f[order]

                if offspring_f[0] < best_f:
                    best_f = offspring_f[0]
                    best_x = offspring[0].copy()

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * offspring[:mu], axis=0)

                # Update evolution paths
                y_diff = (new_mean - x_mean) / (sigma + 1e-300)
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_diff
                ps = (1 - cs_param) * ps + np.sqrt(cs_param * (2 - cs_param) * mueff) * y_diff

                # Update covariance matrix
                delta_hs = 1 + 2 * max(0, np.linalg.norm(ps) / np.sqrt(1 - (1 - cs_param) ** 2) / cs_target - 1)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * np.eye(dim))
                # Efficient weighted outer sum
                diff = (offspring[:mu] - x_mean) / (sigma + 1e-300)
                C += cmu * (weights[:, np.newaxis] * diff) @ diff.T

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = (eigvecs * eigvals) @ eigvecs.T

                # Sigma adaptation (success rate)
                # If more than half the selected offspring are better than mean, increase sigma
                # Use a simple heuristic: compare best offspring to mean
                if offspring_f[0] < np.mean(offspring_f[:mu]):
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                sigma = max(1e-6, min(sigma, 10.0))

                x_mean = new_mean

                # Convergence check
                if sigma < 1e-4 or np.max(np.diag(C)) < 1e-12:
                    break

        # Final local refinement
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            local_sigma = 0.3
            for _ in range(min(30, budget - call_count)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
