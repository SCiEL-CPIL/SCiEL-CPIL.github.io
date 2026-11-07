"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Initial population: center + spread
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # CMA-ES parameters
        # For dim=5, lambda=4*dim is standard
        lam = 4 * dim
        sigma = 3.0  # initial step size
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 + 2 / (dim + 1) - dim - 1) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.3) / (dim + 5)
        ds = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / 4) / (dim + 4)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Weighting for selection
        w = np.log(np.arange(1, lam / 2 + 1)) - np.log(lam / 2)
        w = w / np.sum(w)
        mueff = 1 / np.sum(w ** 2)
        weights = np.concatenate([w, -w])

        # Covariance matrix and inverse
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def update_cma(C, C_inv, C_sqrt, C_inv_sqrt, y, mu, c1, cmu, mueff, dim, lam, w):
            """Update covariance matrix and its factors."""
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - mu) * C) + cmu * sum w_i * y_i * y_i.T
            H = np.zeros((dim, dim))
            for i in range(lam):
                H += w[i] * np.outer(y[i], y[i])
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - mu) * np.eye(dim)) + cmu * H
            # Symmetrize
            C = (C + C.T) / 2
            # Invert and square root
            try:
                C_inv = np.linalg.inv(C)
                C_inv = (C_inv + C_inv.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C_inv)
                eigvals = np.maximum(eigvals, 1e-20)
                C_inv_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                pass
            return C, C_inv, C_sqrt, C_inv_sqrt

        for gen in range(budget // lam):
            if time.time() > deadline - 5:
                break

            # Generate population
            pop = np.zeros((lam, dim))
            pop_f = np.zeros(lam)

            for i in range(lam):
                z = np.random.randn(dim)
                y = C_sqrt @ z
                x = x0 + sigma * y
                x = np.clip(x, -5, 5)
                pop[i] = x
                pop_f[i] = func(x)
                calls += 1
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = x.copy()

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_sorted = pop[idx]
            pop_f_sorted = pop_f[idx]

            # Update step size
            y_sigma = (C_inv_sqrt @ (pop_sorted[:lam // 2].mean(axis=0) - x0)) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update evolution path
            y_mean = (pop_sorted[:lam // 2].mean(axis=0) - x0) / sigma
            pc = (1 - c1) * pc + c1 * y_mean * hsig

            # Update mean
            x0 = x0 + sigma * (w[:lam // 2] @ (pop_sorted[:lam // 2] - x0) / sigma)

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update covariance
            y = (pop_sorted[:lam // 2] - x0) / sigma
            C, C_inv, C_sqrt, C_inv_sqrt = update_cma(
                C, C_inv, C_sqrt, C_inv_sqrt, y, mueff, c1, cmu, mueff, dim, lam, w[:lam // 2]
            )

            # Check for time
            if time.time() > deadline - 10:
                break

        # Use remaining budget for fine-tuning with smaller steps
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline - 2:
            # Local search around best point
            sigma_local = 0.1
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(best_x + sigma_local * np.random.randn(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
