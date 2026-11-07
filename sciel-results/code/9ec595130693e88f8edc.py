"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(0)

        # Initial population size
        N = 4 + int(3 * np.log(dim))  # for dim=5, N=4+3*1.609=8 (rounded)
        N = max(N, 4)

        # CMA-ES parameters
        c_sigma = (N + 2) / (N + 2 + 4 * np.log(N + 2))
        c_c = (4 + N / dim) / (N + 4 + 2 * (N / dim))
        c_1 = 2 / ((N + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((N / 2 + 1) / (N + 2)) ** 2 - 1)

        # Initial sigma
        sigma = 2.0

        # Initial mean
        m = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        inv_sqrt_C = (eigvecs * (1.0 / np.sqrt(np.maximum(eigvals, 1e-20)))) @ eigvecs.T
        D = 1.0 / np.sqrt(np.maximum(eigvals, 1e-20))
        chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        num_calls = 0
        best_x = None
        best_f = np.inf

        def update_eigen():
            nonlocal eigvals, eigvecs, inv_sqrt_C, D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            inv_sqrt_C = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            D = 1.0 / np.sqrt(eigvals)

        while num_calls < budget:
            if time.time() > deadline:
                break

            # Generate N samples
            samples = []
            for _ in range(N):
                if num_calls >= budget:
                    break
                z = rng.standard_normal(dim)
                x = m + sigma * (inv_sqrt_C @ z)
                x = np.clip(x, -5, 5)
                fx = func(x)
                num_calls += 1
                samples.append((fx, x))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(samples) < 2:
                break

            # Sort by fitness
            samples.sort(key=lambda t: t[0])
            fitnesses = np.array([s[0] for s in samples])
            xs = np.array([s[1] for s in samples])

            # Weights
            mu = N // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_mu = min(1 - c_1, 2 * (mu_eff / (dim + 2)) - 1)

            # Update mean
            m_new = m + sigma * np.sum(w[:, None] * xs[:mu], axis=0)
            m_new = np.clip(m_new, -5, 5)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                       (inv_sqrt_C @ ((m_new - m) / sigma))
            chi_norm = np.sqrt(np.sum(p_sigma ** 2))
            h_sigma = 1.0 if chi_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (num_calls + 1) / N)) < \
                       (1.4 + 2 / (dim + 1)) * chi_N else 0.0
            sigma *= np.exp(c_sigma / chi_N * (chi_norm / chi_N - 1))
            sigma = max(sigma, 1e-10)

            # Update C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            h_c = 1.0 if (np.sum(p_c ** 2) / (1 - (1 - c_c) ** (2 * (num_calls + 1) / N))) / \
                    (c_c * (2 - c_c)) < (4 + 2 / (dim + 1)) * chi_N ** 2 else 0.0

            delta_h = (1 - h_c) * c_c * (2 - c_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (xs[i] - m) / sigma
                rank_mu += w[i] * np.outer(diff, diff)

            C = (1 + c_1 * (2 - c_1 - c_mu) + c_1 * delta_h - c_1) * C + \
                c_1 * (np.outer(p_c, p_c) + (1 - h_c) * delta_h * C) + \
                c_mu * rank_mu

            m = m_new

            # Periodically update eigen decomposition
            if num_calls % N == 0:
                update_eigen()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            num_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
