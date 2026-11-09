"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB-style continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = int(4 + 3 * np.log(dim))
        pop_size = max(pop_size, 4)

        # Initial center and sigma
        rng = np.random.default_rng(42)
        x_center = np.zeros(dim)
        sigma = 2.0

        # CMA-ES matrices
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Update rates
        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_c = (4 + dim / pop_size) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cholesky_safe(M):
            try:
                return np.linalg.cholesky(M)
            except:
                # Add small regularization
                M_reg = M + 1e-10 * np.eye(dim)
                return np.linalg.cholesky(M_reg)

        best_x = x_center.copy()
        best_f = func(best_x)
        calls_made = 1

        for gen in range(budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sample from N(x_center, sigma^2 * C)
                z = rng.standard_normal(dim)
                x_i = x_center + sigma * (C_sqrt @ z)
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls_made += 1

                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

                # Store for sorting
                if i == 0:
                    xs = [x_i]
                    fs = [f_i]
                else:
                    xs.append(x_i)
                    fs.append(f_i)

            if len(xs) < mu:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = [xs[i] for i in idx[:mu]]
            fs_sorted = [fs[i] for i in idx[:mu]]

            # Compute weighted mean
            m_new = np.sum([w * x for w, x in zip(weights, xs_sorted)], axis=0)

            # Update evolution paths
            diff = (m_new - x_center) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * diff
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff * c_c * (2 - c_c)) * diff

            # Update covariance
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // pop_size + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            delta_h = 1 - h_sigma
            C = (1 + c_c * (2 - c_c) * delta_h - c_c) * C + \
                c_c * delta_h * np.outer(p_c, p_c) + \
                c_c * (1 - c_c) * np.sum([w * np.outer(x - x_center, x - x_center) / sigma ** 2 for w, x in zip(weights, xs_sorted)], axis=0)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update center
            x_center = m_new
            x_center = np.clip(x_center, -5, 5)

            # Recompute matrix square roots
            C_sqrt = cholesky_safe(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
