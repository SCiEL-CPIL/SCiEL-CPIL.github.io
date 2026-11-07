import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # 50s margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        # Best-ever tracking
        x_ever = np.full(dim, np.nan)
        f_ever = np.inf

        def try_update(x, f):
            nonlocal x_ever, f_ever
            if f < f_ever:
                f_ever = f
                x_ever = x.copy()
            return f < f_ever

        # Phase 1: CMA-ES with large initial sigma to find global basin
        # Lunacek has a shifted global optimum; large sigma helps find it
        x0 = rng.uniform(-4, 4, dim)
        f0 = func(x0)
        calls += 1
        improved = try_update(x0, f0)

        # CMA-ES parameters
        pop_size = 4 * dim + 2  # standard CMA-ES population
        if pop_size > 20:
            pop_size = 20
        mu = pop_size // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 3.0  # Large initial sigma to explore [-5,5]^5 broadly
        x_mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Restart logic
        stall_count = 0
        stall_threshold = 30  # generations without improvement
        max_stalls = 3  # total restarts allowed

        while calls + pop_size <= budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            B = rng.standard_normal((pop_size, dim))
            # Whitening transform (use Cholesky of C)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            offspring = x_mean + sigma * (B @ L.T)
            offspring = np.clip(offspring, -5, 5)

            # Evaluate
            f_vals = np.zeros(pop_size)
            for i in range(pop_size):
                if time.time() > deadline - 3.0:
                    break
                f_vals[i] = func(offspring[i])
                calls += 1
            else:
                # All evaluated
                pass
            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(f_vals)
            y = offspring[order]
            f_sorted = f_vals[order]

            # Update best
            if f_sorted[0] < f_ever:
                stall_count = 0
            else:
                stall_count += 1

            try_update(y[0], f_sorted[0])

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * y[:mu], axis=0)

            # Update evolution paths
            y_inv = np.linalg.solve(L, (x_mean - x_mean_old) / (sigma * chi_n))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv
            norm_ps = np.linalg.norm(ps)
            h_sig = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / (dim + 1))) < (1 + 2 / (dim + 1)) * chi_n else 0.0

            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma
            C = (1 + c_1 * h_sig * (mu_eff - 1) - c_1) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(weights[:, None, None] * np.outer(y[:mu] - x_mean_old, y[:mu] - x_mean_old), axis=0) / sigma ** 2

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Restart if stalled
            if stall_count > stall_threshold and max_stalls > 0:
                max_stalls -= 1
                stall_count = 0
                # Restart from a new random point, keep best-ever
                x_mean = rng.uniform(-4, 4, dim)
                f_new = func(x_mean)
                calls += 1
                try_update(x_mean, f_new)
                sigma = 3.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                if calls >= budget:
                    break

        # Phase 2: Fine-tune with small steps using remaining budget
        if calls < budget:
            sigma_small = max(sigma * 0.1, 1e-4)
            while calls < budget:
                if time.time() > deadline - 2.0:
                    break
                y = x_mean + sigma_small * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < f_ever:
                    x_mean = y.copy()
                    try_update(y, fy)
                    sigma_small *= 1.1
                else:
                    sigma_small *= 0.9
                    sigma_small = max(sigma_small, 1e-8)

        return x_ever, f_ever
