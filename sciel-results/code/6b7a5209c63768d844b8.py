"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive sigma and population for Rosenbrock-like problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        c_sigma = (lam + 2) / ((lam + 5) * (lam + 5))
        c_c = (4 + lam / dim) / ((dim + 4) * (lam + 16))
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) ** 2 - 1))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        rng = np.random.default_rng(12345)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x_mean.copy()
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        best_f = eval_func(x_mean)

        # Precompute B and D from C
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.clip(eigvals, 1e-12, None))
        B = eigvecs

        gen = 0
        while calls + lam <= budget and time.time() < deadline:
            # Sample offspring
            off = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                off.append(x_mean + sigma * (B @ (D * z)))

            off_f = np.array([eval_func(y) for y in off])
            idx = np.argsort(off_f)
            off = [off[i] for i in idx]

            # Recombine
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * off[i]

            # Update paths
            x_old = x_mean.copy()
            x_mean = x_new

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_mean - x_old)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (x_mean - x_old)))

            # Adapt sigma
            delta_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Adapt covariance
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc + (1 - delta_sig) * c_c * (2 - c_c) * np.outer(pc, pc))
            for i in range(mu):
                v = B @ (D * (off[i] - x_old)) / sigma
                C += c_mu * w[i] * np.outer(v, v)

            # Re-orthogonalize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
