"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_run(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES for a given budget from initial point x0 with step sigma0."""
        n = self.dim
        x = np.array(x0, dtype=float)
        f = func(x)
        sigma = sigma0
        calls = 1

        # CMA-ES parameters
        lambda_ = max(4, 4 + int(3 * np.log(n)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # Precompute C^0.5 initially
        C_sqrt = np.eye(n)
        # We'll update C_sqrt via eigen-decomposition when needed

        best_x = x.copy()
        best_f = f

        remaining = budget - calls
        while remaining > 0 and time.time() < deadline:
            # Generate population
            Y = []
            F = []
            for i in range(min(lambda_, remaining)):
                # Sample from N(x, sigma^2 * C)
                # Use Cholesky-like update: we maintain C_sqrt
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                Y.append(y)
                F.append(fy)
                calls += 1
                remaining -= 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(Y) == 0:
                break

            Y = np.array(Y)
            F = np.array(F)

            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]

            # Update mean
            x_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update step size
            # Compute h_c for delta correction
            C_sqrt_old = C_sqrt
            # Approximate: use norm of (x_new - x) / sigma
            delta = np.linalg.norm(x_new - x) / sigma if sigma > 1e-15 else 0
            h_c = 1.0 if delta < (1.4 + 2 / (n + 1)) * chi_n else 0.0

            p_sigma = (1 - c_sigma) * p_sigma + h_c * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                C_sqrt_old @ ((x_new - x) / sigma)
            ) if sigma > 1e-15 else p_sigma
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            y_w = (Y_sorted[:mu] - x) / sigma
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(y_w, y_w), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update C_sqrt via eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            x = x_new
            f = np.min(F_sorted)

            if f < best_f:
                best_f = f
                best_x = Y_sorted[0].copy()

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 5 second safety margin
        rng = np.random.default_rng(42)
        n = self.dim

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Multi-start random search to find a good initial point
        # Use 20% of budget for exploration
        explore_budget = max(20, self.budget // 5)
        for _ in range(explore_budget):
            if time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls += 1

        # Phase 2: CMA-ES from the best point found
        remaining = self.budget - total_calls
        if remaining > 5 and best_x is not None:
            # Start CMA-ES with a reasonable initial sigma
            sigma0 = 1.0
            x_opt, f_opt, calls_used = self._cma_run(
                func, best_x, sigma0, remaining, rng, deadline
            )
            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: If we have time/budget left, do another CMA-ES run from a perturbed best
        remaining = self.budget - total_calls
        if remaining > 10 and best_x is not None and time.time() < deadline:
            # Perturb best solution
            x_perturb = best_x + 0.5 * rng.standard_normal(n)
            x_perturb = np.clip(x_perturb, -5, 5)
            x_opt2, f_opt2, calls_used2 = self._cma_run(
                func, x_perturb, 0.5, remaining, rng, deadline
            )
            total_calls += calls_used2
            if f_opt2 < best_f:
                best_f = f_opt2
                best_x = x_opt2.copy()

        # Final: ensure we haven't exceeded budget
        # Return the best solution found
        return best_x, best_f
# EVOLVE-BLOCK-END
