"""CMA-ES optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance matrix adaptation for rotated/shifted landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size: lambda = 4 + floor(3 * log(n))
        lam = max(4, int(4 + 3 * np.log(max(n, 2))))
        # For n=5: lam = 4 + floor(3*1.609) = 4 + 4 = 8
        # Use slightly larger for better exploration
        lam = max(lam, 10)

        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 1.3) ** 2 + 4)
        cs = (n + 1.7) / (n + n + 5.5)
        cmu = min(1 - 2 * cc, 0.5 * (n - 2) / (n + 2))
        # Standard chi_n for expected ||N(0,I)||
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)

        best_x = m.copy()
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = float(func(x_clipped))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Initial evaluation
        evaluate(m)

        # Generate initial population to get a good starting sigma estimate
        # Actually, let's just start CMA-ES directly

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Determine how many generations we can afford
            remaining = budget - calls
            if remaining < lam:
                # Not enough for a full generation, do single step
                step = sigma * rng.multivariate_normal(np.zeros(n), C)
                x_new = np.clip(m + step, -5.0, 5.0)
                f_new = evaluate(x_new)
                if f_new < best_f:
                    m = x_new.copy()
                sigma *= 0.87
                break

            # Generate population
            # Decompose C = B diag(D^2) B^T
            # We maintain B and D such that C = B @ diag(D**2) @ B.T
            # Actually, standard CMA-ES maintains C directly and does Cholesky
            # Let's use the standard approach with C matrix

            # Generate samples: z_i ~ N(0, I), x_i = m + sigma * B @ diag(D) @ z_i
            # But we need to maintain B and D from C

            # Recompute B, D from C via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip to positive
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Generate population
            Y = []
            X = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = B @ (D * z)
                x = np.clip(m + sigma * y, -5.0, 5.0)
                Y.append(y)
                X.append(x)

            # Evaluate population
            F = [evaluate(x) for x in X]

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = [X[i] for i in order]
            F_sorted = [F[i] for i in order]
            Y_sorted = [Y[i] for i in order]

            # Update mean
            mu = lam // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            m_old = m.copy()
            m = sum(w * X_sorted[i] for i, w in enumerate(weights))

            # Update sigma via cs
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / D @ B.T @ (m - m_old) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / D @ B.T @ (m - m_old) / sigma

            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = sum(
                weights[i] * np.outer(Y_sorted[i] / D, Y_sorted[i] / D)
                for i in range(mu)
            )
            C = (1 + mu_eff * (cc - 1) - cc) * C + cc * rank_one + \
                (1 + mu_eff * (cs - 1) - cs) * 0 + cmu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(n)

            # Update sigma
            sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Check for convergence
            if sigma < 1e-12:
                break

            # If no progress in last few generations, consider restart
            # For simplicity, we'll just continue

        return best_x, best_f
