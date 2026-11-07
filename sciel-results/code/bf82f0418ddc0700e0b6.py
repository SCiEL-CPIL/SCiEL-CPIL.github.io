"""Rewritten optimizer using CMA-ES (pure Python/NumPy) with multi-start and adaptive restart."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based derivative-free optimizer with multi-start diversification."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 285  # safety margin before 300s limit
        self._call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def cma_es(start_x, sigma_init, max_iter):
            nonlocal deadline
            dim = self.dim
            m = start_x.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_inv = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            # Strategy parameters
            mu = max(1, dim // 2)  # number of recombination weights
            lambda_ = 2 * mu  # population size
            w = np.zeros(mu)
            w_sum = np.sum(np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)))
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w /= w_sum
            mu_eff = 1.0 / np.sum(w ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Cholesky decomposition helper
            def chol_update(C_new):
                nonlocal C, C_inv, B, D
                try:
                    L = np.linalg.cholesky(C_new)
                    D = np.sqrt(np.diag(L))
                    B = L / D[:, np.newaxis]
                    C = C_new
                    C_inv = np.linalg.inv(C_new)
                except np.linalg.LinAlgError:
                    # Fallback: eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C_new)
                    eigvals = np.maximum(eigvals, 1e-12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                    C = C_new
                    C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

            chol_update(C)

            for gen in range(max_iter):
                if time.time() > deadline or self._call_count >= self.budget:
                    break

                # Generate population
                ys = np.zeros((lambda_, dim))
                xs = np.zeros((lambda_, dim))
                fs = np.zeros(lambda_)

                for i in range(lambda_):
                    z = np.random.standard_normal(dim)
                    y = B @ (D * z)
                    ys[i] = y
                    xs[i] = m + sigma * y
                    fs[i] = counted_func(xs[i])
                    if self._call_count >= self.budget:
                        return

                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = xs[order]
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update mean
                m_old = m.copy()
                m = np.sum(w[:, np.newaxis] * xs_sorted[:mu], axis=0)

                # Update step-size control
                y_mean = np.sum(w[:, np.newaxis] * ys_sorted[:mu], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean / np.linalg.norm(B.T @ D)
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt((mu_eff + 1) / (dim + 1)) - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update evolution path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (m - m_old) / sigma)

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(ys_sorted[i], ys_sorted[i])

                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * (B @ rank_mu @ B.T)

                # Symmetrize
                C = (C + C.T) / 2

                # Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals)) * np.eye(dim)

                # Update Cholesky
                chol_update(C)

                # Check for convergence
                if np.all(np.abs(ys_sorted[:mu]) < 1e-10) or sigma < 1e-15:
                    break

            return m, fs_sorted[0]

        # Multi-start with diverse initial points
        rng = np.random.default_rng(42)

        # Define starting points
        starts = []
        sigmas = []

        # Center
        starts.append(np.zeros(self.dim))
        sigmas.append(2.0)

        # Random points in the box
        for _ in range(7):
            starts.append(rng.uniform(-4, 4, self.dim))
            sigmas.append(1.5)

        # Allocate budget
        total_starts = len(starts)
        base_alloc = self.budget // total_starts
        base_alloc = max(base_alloc, 20)

        for i, (start, sigma) in enumerate(zip(starts, sigmas)):
            if self._call_count >= self.budget or time.time() > deadline:
                break

            # Allocate more iterations to earlier starts
            if i == 0:
                max_iter = base_alloc * 2
            elif i < total_starts // 2:
                max_iter = base_alloc
            else:
                max_iter = base_alloc // 2

            cma_es(start, sigma, max_iter)

        return best_x, best_f
# EVOLVE-BLOCK-END
