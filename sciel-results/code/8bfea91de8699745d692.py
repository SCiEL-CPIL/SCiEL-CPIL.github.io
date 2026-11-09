"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with restarts for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es_restart(self, func, x_start, sigma_start, budget, rng):
        """Run one CMA-ES restart, returning best_x, best_f, calls_made."""
        n = self.dim
        lambda_ = 2 * n
        mu = lambda_ // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (n + 2)) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        x = x_start.copy()
        sigma = sigma_start
        ps = np.zeros(n)
        pc = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)

        fx = func(x)
        calls = 1
        best_f = fx
        best_x = x.copy()

        for gen in range(budget - 1):
            if calls >= budget:
                break

            # Generate and evaluate offspring
            ys = np.zeros((lambda_, n))
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (D * (B @ z))
                y = np.clip(y, -5, 5)
                ys[i] = y
                fvals[i] = func(y)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = y.copy()

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            ys_sorted = ys[idx]
            fvals_sorted = fvals[idx]

            # Update mean
            x_old = x.copy()
            x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update step size control vectors
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * D) * (B @ (x_new - x_old))
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff)
            if hsig:
                pc = (1 - c_c) * pc + c_c / sigma * (x_new - x_old)

            # Update covariance
            delta_d = (2 * hsig - 1) * c_c * (1 - c_c)
            rank_1 = np.outer(pc, pc)
            diff = ys_sorted[:mu] - x_old
            rank_mu = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Eigendecomposition
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-20))
                D = D2
                B = B2
            except np.linalg.LinAlgError:
                pass

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new

        return best_x, best_f, calls

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        remaining = total_budget

        # Restart strategy: (budget_fraction, initial_sigma)
        restarts = [
            (50, 2.0),
            (30, 1.0),
            (20, 0.5),
        ]

        for frac, sigma_init in restarts:
            if remaining <= 0:
                break
            b = int(frac / 100 * total_budget)
            b = min(b, remaining)
            if b < 10:
                b = remaining
            if b < 1:
                break
            remaining -= b

            if best_x is not None:
                x_start = best_x + rng.normal(0, 0.3, n)
                x_start = np.clip(x_start, -5, 5)
            else:
                x_start = rng.uniform(-5, 5, n)

            bx, bf, calls = self._cma_es_restart(func, x_start, sigma_init, b, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx

        # Use any remaining budget for local search around best
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
            if fx < best_f:
                best_f = fx
                best_x = x

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
