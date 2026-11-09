"""Optimizer for BBOB f6 (Attractive Sector) using CMA-ES with adaptive strategy."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES based optimizer with global exploration and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, deadline):
        """
        Run CMA-ES from x0 with initial step size sigma0.
        Returns (best_x, best_f, calls_used).
        """
        n = self.dim
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Mean update
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # Cholesky of C
        C_chol = np.eye(n)

        calls_used = 0
        best_x = x_mean.copy()
        best_f = float(func(x_mean))
        calls_used += 1

        # Track best
        archive = [(best_f, best_x.copy())]

        for gen in range(max_calls // lambda_ + 1):
            if calls_used >= max_calls or time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls_used >= max_calls or time.time() > deadline:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(n)
                # Use Cholesky of C
                x_off = x_mean + sigma * (C_chol @ z)
                x_off = np.clip(x_off, -5.0, 5.0)
                f_off = float(func(x_off))
                calls_used += 1
                offspring.append((f_off, x_off))

            if len(offspring) < 2:
                break

            # Sort offspring
            offspring.sort(key=lambda x: x[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
                archive.append((best_f, best_x.copy()))

            # Select parents
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * offspring[i][1]
            y_w = y_w / sigma

            # Update mean
            x_mean = x_mean + c_c * (y_w - x_mean) + c_1 * (y_w - x_mean)
            # Actually, standard CMA-ES:
            # x_mean = x_mean + c_1 * (y_w - x_mean) + c_mu * sum(w_i * (x_i - x_mean_old)/sigma)
            # Let's redo this properly
            x_mean_old = x_mean - c_c * (y_w - x_mean) - c_1 * (y_w - x_mean)
            # Hmm, let me restructure. Let me use a cleaner implementation.
            # Actually let me restart with a cleaner CMA-ES.
            break

        return best_x, best_f, calls_used

    def _cmaes_clean(self, func, x0, sigma0, max_calls, deadline):
        """
        Clean CMA-ES implementation.
        """
        n = self.dim
        rng = np.random.default_rng(54321)

        # Parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_chol = np.eye(n)

        calls_used = 0
        best_x = x_mean.copy()
        best_f = float(func(x_mean))
        calls_used += 1

        for gen in range(1, max_calls // lambda_ + 2):
            if calls_used >= max_calls or time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls_used >= max_calls or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_off = x_mean + sigma * (C_chol @ z)
                x_off = np.clip(x_off, -5.0, 5.0)
                f_off = float(func(x_off))
                calls_used += 1
                offspring.append((f_off, x_off))

            if len(offspring) < 2:
                break

            offspring.sort(key=lambda x: x[0])

            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Weighted sum of selected points
            y_w = np.zeros(n)
            for i in range(min(mu, len(offspring))):
                y_w += weights[i] * offspring[i][1]
            y_w = y_w / sigma

            # Update mean
            x_mean = x_mean + (c_1 + c_mu) * y_w

            # Update p_sigma
            norm_y_w = np.linalg.norm(y_w)
            h_sigma = 1.0 if norm_y_w / np.sqrt(1 - (1 + gen / 10) * (1 - c_sigma)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(y_w, y_w)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C + c_1 * (outer - C)
            # Wait, the rank-mu update is:
            # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * outer + c_mu * sum(w_i * outer_i)
            # Let me redo
            C = (1 - c_1 - c_mu + delta_h) * C
            C += c_1 * outer
            for i in range(min(mu, len(offspring))):
                y_i = (offspring[i][1] - (x_mean - (c_1 + c_mu) * y_w)) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition for B and D
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                C_chol = B @ np.diag(D)
            except np.linalg.LinAlgError:
                # Fallback: reset C
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                C_chol = np.eye(n)

            # Clip x_mean to box
            x_mean = np.clip(x_mean, -5.0, 5.0)

            # Stop if sigma is too small
            if sigma < 1e-10:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Global scan to find a good starting point
        n_scan = 30
        for i in range(n_scan):
            if time.time() > deadline or calls_used >= budget:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = np.ones(dim) * 2.0
            elif i == 2:
                x = np.ones(dim) * -2.0
            else:
                x = rng.uniform(-5.0, 5.0, dim)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES refinement from best point
        remaining = budget - calls_used
        if remaining > 20 and time.time() < deadline:
            sigma0 = 1.0  # Initial step size
            x_cma, f_cma, used = self._cmaes_clean(func, best_x, sigma0, remaining, deadline)
            calls_used += used
            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma.copy()

        # Phase 3: If we have more budget, do another CMA-ES with smaller sigma
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline:
            sigma0 = 0.1
            x_cma2, f_cma2, used = self._cmaes_clean(func, best_x, sigma0, remaining, deadline)
            calls_used += used
            if f_cma2 < best_f:
                best_f = f_cma2
                best_x = x_cma2.copy()

        # Phase 4: Final local search with very small sigma
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline:
            sigma0 = 0.01
            x_cma3, f_cma3, used = self._cmaes_clean(func, best_x, sigma0, remaining, deadline)
            calls_used += used
            if f_cma3 < best_f:
                best_f = f_cma3
                best_x = x_cma3.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
