"""CMA-ES portfolio optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with time guard for 5D BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters for dim=5
        lam = 4 + int(3 * np.log(dim))  # = 9 for dim=5
        mu = lam // 2  # = 4
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0, using at most max_evals function calls."""
            nonlocal deadline
            x = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            hsig = (1 + 2.0 / max(dim - 2, 1)) * (1 + cc) * cs * chi

            evals = 0
            best_x = x.copy()
            best_f = float('inf')

            # Evaluate initial point
            fx = func(x)
            evals += 1
            best_f = fx
            best_x = x.copy()

            for gen in range(max_evals // lam + 2):
                if time.time() > deadline:
                    break
                if evals >= max_evals:
                    break

                # Compute B and D from C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Generate population
                pop = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    pop[i] = x + sigma * (B @ (D * z))
                    pop[i] = np.clip(pop[i], -5, 5)

                # Evaluate population
                fvals = np.full(lam, float('inf'))
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    fvals[i] = func(pop[i])
                    evals += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

                # Sort
                idx = np.argsort(fvals)
                pop_sorted = pop[idx]
                fvals_sorted = fvals[idx]

                # Mean update
                x_new = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
                y = x_new - x
                x = x_new

                # Path updates
                p_sigma = (1 - cs) * p_sigma + cs * damps * y / sigma
                p_c = (1 - cc) * p_c + cc * damps * y / sigma

                # hsig test
                norm_pc = np.linalg.norm(p_c)
                hsig_val = norm_pc / np.sqrt(1 - (1 - cc) ** (gen + 1)) / chi
                delta_d = float(hsig_val < (1.4 + 2 / (dim + 1)) * hsig)

                # Covariance update
                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(y, y) + (1 - delta_d) * cc * (2 - cc) * C)
                for i in range(mu):
                    yi = pop_sorted[i] - x_new
                    C += cmu * w[i] * np.outer(yi, yi)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals_c, _ = np.linalg.eigh(C)
                if np.min(eigvals_c) < 1e-10:
                    C += 1e-10 * np.eye(dim)

                # Step size update
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp(cs / damps * (norm_p_sigma / chi - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

            return best_x, best_f

        # Run CMA-ES
        x0 = rng.uniform(-4, 4, dim)
        sigma0 = 1.5
        best_x, best_f = cma_es(x0, sigma0, budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
