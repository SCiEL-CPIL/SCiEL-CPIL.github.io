import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size
        lam = max(4 + int(3 * np.log(n)), 10)
        mu = lam // 2

        # Log-linear weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)

        # Initial values
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Learning rates
        cc = (4 + mueff / n) / ((n + 4) * (2 + mueff / n))
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        # Damping
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        hsig_threshold = (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) * cc * (2 - cc))

        # Track best
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1

        # C^{-1/2} and Cholesky factor
        eigvals, eigvecs = np.linalg.eigh(C)
        Csqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        chol = np.linalg.cholesky(C)

        gen = 0
        while calls < budget:
            gen += 1

            # Generate population
            x_pop = np.zeros((lam, n))
            f_pop = np.zeros(lam)

            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                x = m + sigma * (chol @ z)
                x = np.clip(x, -5, 5)
                x_pop[i] = x
                f_pop[i] = func(x)
                calls += 1
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = x.copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]

            # Update mean
            m_new = m + sigma * np.sum(w[:mu, None] * (x_sorted[:mu] - m), axis=0)
            y = (m_new - m) / sigma if sigma > 1e-10 else np.zeros(n)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Csqrt_inv @ y)
            ps_norm = np.linalg.norm(ps)
            hsig = ps_norm / np.sqrt(1 - (1 - cs)**(2 * gen)) < hsig_threshold
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y

            # Update mean
            m = m_new

            # Update sigma
            sigma *= np.exp(cs / damps * (ps_norm / hsig_threshold - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update covariance matrix
            yw = (x_sorted[:mu] - m) / sigma
            delta_sigma = pc * pc.T + (1 - hsig) * cc * (2 - cc) * C
            delta_mu = np.sum(w[:mu, None] * (yw * yw.T), axis=0)

            C = (1 - c1 - cmu) * C + c1 * delta_sigma + cmu * delta_mu
            C = (C + C.T) / 2

            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C^{-1/2} and Cholesky
            eigvals, eigvecs = np.linalg.eigh(C)
            Csqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            try:
                chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(n)
                chol = np.linalg.cholesky(C)

        return best_x, best_f
