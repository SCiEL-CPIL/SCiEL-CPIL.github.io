"""Global-to-local optimizer for BBOB f6: random exploration then CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global random search followed by CMA-ES local optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s wall-clock limit

        dim = self.dim
        total_budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None

        rng = np.random.default_rng(123)

        # Phase 1: Global exploration with 50 uniform random evaluations
        n_explore = 50
        explore_points = []
        explore_vals = []

        for _ in range(n_explore):
            if n_calls >= total_budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            n_calls += 1
            explore_points.append(x)
            explore_vals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        explore_vals = np.array(explore_vals)
        explore_points = np.array(explore_points)

        # Pick the best point as starting point for CMA-ES
        best_idx = np.argmin(explore_vals)
        x0 = explore_points[best_idx].copy()

        # Phase 2: CMA-ES on remaining budget
        remaining = total_budget - n_calls
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        lambda_pop = 12
        max_evals_cma = remaining

        x = x0.copy()
        sigma = 2.0  # initial step size
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # CMA-ES parameters
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        gen = 0
        while n_calls < total_budget:
            if time.time() > deadline:
                break

            # Check if we have enough budget for a full population
            if n_calls + lambda_pop > total_budget:
                # Use remaining budget for partial population
                pop_size = total_budget - n_calls
                if pop_size < 2:
                    break
                # For partial population, just do simple random search around best
                for _ in range(pop_size):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    n_calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                break

            # Generate population
            ys = []
            fs = []
            for j in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                n_calls += 1
                ys.append(y)
                fs.append(fy)

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update global best
            for j in range(min(mu, len(fs))):
                if fs[j] < best_f:
                    best_f = fs[j]
                    best_x = ys[j].copy()

            # Weighted mean
            m_new = np.sum(weights[:, None] * ys[:mu], axis=0)

            # Update CMA matrix
            y_sigma = (m_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n) < (1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * c_c * y_sigma
            delta_hsig = (1 - hsig) * c_c * (1 - c_c)

            v_new = np.zeros((dim, dim))
            for i in range(mu):
                v_new += weights[i] * np.outer(ys[i] - m_new, ys[i] - m_new)
            C = (1 + c_1 * delta_hsig + c_mu - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_hsig) \
                + c_mu * v_new / (sigma ** 2)

            # Ensure positive definite
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ys[0] - m_new) / sigma - 1))
            sigma = np.clip(sigma, 0.001, 10.0)

            # Move center
            x = m_new
            gen += 1

            # Stagnation check: if no improvement in last 30 evaluations, restart CMA-ES
            # from the best point found so far with a fresh sigma
            if gen > 0 and gen % 30 == 0:
                # Check if best has improved in last 30 evals (approximate)
                pass  # Keep it simple; CMA-ES's own adaptation handles this

        return best_x, best_f
# EVOLVE-BLOCK-END
