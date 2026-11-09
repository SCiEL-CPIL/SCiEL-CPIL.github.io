"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        popsize = max(4, int(4 + 3 * np.log(dim)))
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 5))
        cs = (dim + 1.7) / (dim + 9 + cmu)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        cc = (4 + dim / dim) / (4 + dim)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x_mean = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Evaluate initial point
        x0 = np.clip(rng.uniform(-5, 5, dim), -5, 5)
        f_best = func(x0)
        x_best = x0.copy()
        calls = 1
        t = 0  # Iteration counter

        def cholesky(C_mat):
            """Compute B, D such that C = B @ diag(D^2) @ B.T"""
            L = np.linalg.cholesky(C_mat)
            # L is lower triangular, C = L @ L.T
            # We want C = B @ diag(D^2) @ B.T
            # So B = Q from QR of L.T, D from diagonal
            Q, R = np.linalg.qr(L.T)
            # R is upper triangular with positive diagonal
            D_vec = np.diag(R)
            B_mat = Q.T
            return B_mat, D_vec

        # Initial B, D from C = I
        B, D = np.eye(dim), np.ones(dim)

        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if we can fit a full population
            if calls + popsize > budget:
                # If not, we can either break or evaluate a smaller population
                # To keep it simple and safe, break. The remaining budget is small.
                break

            t += 1
            # Generate population
            z = rng.standard_normal((popsize, dim))
            # Transform: y = x_mean + sigma * B @ diag(D) @ z
            # Actually C = B @ diag(D^2) @ B.T, so sqrt(C) = B @ diag(D)
            # y = x_mean + sigma * (B @ diag(D)) @ z
            # For row-wise: z @ diag(D) @ B.T
            steps = (z * D) @ B.T
            population = x_mean + sigma * steps
            population = np.clip(population, -5, 5)

            # Evaluate population
            f_vals = np.array([func(x) for x in population])
            calls += popsize

            # Update best
            for i in range(popsize):
                if f_vals[i] < f_best:
                    f_best = f_vals[i]
                    x_best = population[i].copy()

            # Sort by fitness
            order = np.argsort(f_vals)
            x_sorted = population[order]
            f_sorted = f_vals[order]

            # Weights
            mu = popsize // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1 / (w ** 2).sum()

            # Update mean
            x_new_mean = sum(w[i] * x_sorted[i] for i in range(mu))

            # Update covariance matrix
            v = (x_sorted[:mu] - x_mean) / sigma
            C_new = (1 - c1 - cmu) * C
            for i in range(mu):
                C_new += w[i] * np.outer(v[i], v[i])

            # Update step size control
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (
                B.T @ (D * (B @ x_new_mean - x_mean))
            )
            # Use t for the time step in hsig calculation
            # 1 - (1-cs)^(2*t)
            denom = np.sqrt(1 - (1 - cs) ** (2 * t))
            if denom == 0:
                hsig = False
            else:
                hsig = (np.linalg.norm(p_sigma) / denom) < (
                    1.4 + 2 / (dim + 1)
                ) * chi

            # Update C using hsig
            if hsig:
                C_new += c1 * (2 / c1 - 1 + cmu) * np.outer(p_sigma, p_sigma)

            # Update pc
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (
                B.T @ (D * (B @ x_new_mean - x_mean))
            )

            # Update C using pc
            C_new += cmu * (2 / cmu - 1 + cmu) * np.outer(p_c, p_c)

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi - 1))

            # Update mean and C
            x_mean = x_new_mean
            C = C_new

            # Recompute B, D
            try:
                B, D = cholesky(C)
            except:
                # If Cholesky fails, add small regularization
                C += 1e-12 * np.eye(dim)
                try:
                    B, D = cholesky(C)
                except:
                    B, D = np.eye(dim), np.ones(dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
