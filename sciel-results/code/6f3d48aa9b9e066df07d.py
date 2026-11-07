# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """IPOP-CMA-ES optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280  # Safety margin before 330s kill

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initial point: center of box
        x = np.zeros(n)
        fx = func(x)
        self.calls += 1
        self.best_x = x.copy()
        self.best_f = fx

        # CMA-ES parameters
        sigma = 2.0
        pop_size = max(4, int(4 + 3 * np.log(n)))  # Initial population size
        c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) * (pop_size + 1.3))
        cmu = min(0.3, (2 + 2.0 / (n + 1.3) - c1) / ((n + 2) * (pop_size + 2.3)))
        cs = (pop_size + 1.7) / (n + pop_size + 8.2)
        c1 = 1 / (n * n) + c1
        c_sigma = cs / (n + 4.5)
        c_cov = (5 + 2 / (n + 1) - c1 - cmu) / (n + n)

        C = np.eye(n)
        Cinv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        hsig = (1 + 6 / (n + 1)) / (1 + c_sigma)
        dcor = n + 2 + 2 * max(0, np.sqrt((1 + c1 / c_cov) * (1 - c1 / c_cov) / max(1e-10, c_sigma)) - 1)
        dcor = max(1.0, dcor)

        best_x = x.copy()
        best_f = fx
        no_improve = 0
        pop_size_base = pop_size
        restart_count = 0

        while self.calls < budget and time.time() < self.deadline:
            # Adaptive population size (IPOP)
            if no_improve > 10 and pop_size < 4 * pop_size_base and self.calls + pop_size < budget:
                pop_size = min(2 * pop_size, 4 * pop_size_base, budget - self.calls)
                no_improve = 0
                # Reset some CMA state for fresh start with larger pop
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                Cinv = np.eye(n)
                sigma = max(sigma * 0.7, 0.5)

            # Generate population
            # Compute inverse square root of C
            try:
                D = np.linalg.cholesky(C)
                D_inv_sqrt = np.linalg.inv(D.T)
            except np.linalg.LinAlgError:
                D_inv_sqrt = np.eye(n)

            samples = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                y = x + sigma * D_inv_sqrt @ z
                y = np.clip(y, -5, 5)
                samples.append(y)

            # Evaluate population
            vals = []
            for y in samples:
                if self.calls >= budget or time.time() > self.deadline:
                    break
                fy = func(y)
                self.calls += 1
                vals.append((fy, y))

            if not vals:
                break

            # Sort by fitness
            vals.sort(key=lambda t: t[0])
            f_sorted = np.array([v[0] for v in vals])
            x_sorted = np.array([v[1] for v in vals])

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                self.best_x = best_x.copy()
                self.best_f = best_f
                no_improve = 0
            else:
                no_improve += 1

            # Update CMA-ES state
            mu_eff = pop_size // 2
            w = np.zeros(pop_size)
            weights = np.exp(-1.0 * np.arange(pop_size) / (pop_size / 2.0))
            weights = weights / weights.sum()
            w[:mu_eff] = weights[:mu_eff]
            mu_star = np.sum(w ** 2)
            w = w / np.sum(w)  # Normalize

            # New center
            x_new = np.sum(w[:, None] * x_sorted[:mu_eff], axis=0)

            # Update evolution path
            diff = (x_new - x) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_star) * diff
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (self.calls + 1) / pop_size)) < (1.4 + 2 / (n + 1)) * chi_n:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_star) * D_inv_sqrt @ diff

            # Update covariance matrix
            delta_sig = (np.linalg.norm(ps) / chi_n - (1 - c_sigma)) ** 2
            C = (1 + c_cov * delta_sig - c_cov) * C
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu_eff):
                rank_mu += w[i] * np.outer(x_sorted[i] - x_new, x_sorted[i] - x_new)
            C += c1 * rank_one + cmu * rank_mu

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update inverse
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                Cinv = np.eye(n)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            # Update center
            x = x_new

            # Check for restart
            if no_improve > 30 and self.calls + 10 < budget:
                # Restart from best with larger sigma
                x = best_x.copy()
                sigma = max(sigma * 1.5, 1.0)
                pop_size = pop_size_base
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                Cinv = np.eye(n)
                no_improve = 0
                restart_count += 1
                if restart_count > 5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
