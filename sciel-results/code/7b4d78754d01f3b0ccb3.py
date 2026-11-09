"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio-based continuous optimizer with CMA-ES and multi-start (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill
        n_calls = 0
        dim = self.dim
        budget = self.budget

        def call_f(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        # Best solution tracking
        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time()

        # --- Phase 1: Global exploration with structured sampling ---
        # Use a combination of Latin hypercube and random points
        n_explore = min(60, budget // 8)

        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_explore)
        lhs_points = lhs_points * 10.0 - 5.0  # map to [-5, 5]

        # Add some structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.ones(dim) * 0.5,
            -np.ones(dim) * 0.5,
        ]
        for i in range(4):
            structured.append(np.random.uniform(-4, 4, dim))

        candidates = np.vstack([lhs_points, np.array(structured)])

        for i in range(len(candidates)):
            if n_calls >= budget or time_left() < 60:
                break
            x = np.clip(candidates[i], -5, 5)
            f = call_f(x)
            if f is not None:
                update_best(x, f)

        # --- Phase 2: CMA-ES from the best point ---
        # Implement a lightweight CMA-ES

        def run_cma_es(x0, f0, n_steps, sigma_init=0.5):
            nonlocal n_calls, best_x, best_f
            dim_local = dim
            lam = max(4, int(4 + 3 * np.log(dim_local)))  # population size
            mu = lam // 2
            weights = np.log(np.arange(mu, 0, -1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim_local + mueff + 5)
            c_c = (4 + mueff / dim_local) / (dim_local + 4 + 2 * mueff / dim_local)
            c_1 = 2 / ((dim_local + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim_local + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim_local) * (1 - 1.0 / (4 * dim_local) + 1.0 / (21 * dim_local ** 2))

            x_mean = x0.copy()
            sigma = sigma_init
            ps = np.zeros(dim_local)
            pc = np.zeros(dim_local)
            C = np.eye(dim_local)
            Cinv_sqrt = np.eye(dim_local)
            # Precompute eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

            for step in range(n_steps):
                if n_calls >= budget or time_left() < 30:
                    break

                # Generate offspring
                ys = []
                fs = []
                for i in range(lam):
                    z = rng.standard_normal(dim_local)
                    y = x_mean + sigma * (Cinv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = call_f(y)
                    if fy is None:
                        return x_mean, f0
                    ys.append(y)
                    fs.append(fy)

                ys = np.array(ys)
                fs = np.array(fs)

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update best
                if fs_sorted[0] < best_f:
                    best_x = ys_sorted[0].copy()
                    best_f = fs_sorted[0]

                # Update mean
                y_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
                # Correction for clipping
                y_mean_clipped = np.clip(y_mean, -5, 5)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (Cinv_sqrt @ (y_mean - x_mean)) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (step + 1)))) < (1.4 + 2 / (dim_local + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (y_mean - x_mean) / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = ((1 + c_1 + c_mu - c_1 - c_c) * C +
                     c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * np.outer(pc, pc) - np.eye(dim_local)) +
                     c_mu * np.sum(weights[:, None] * np.outer(ys_sorted[:mu] - x_mean, ys_sorted[:mu] - x_mean), axis=0))

                # Ensure positive definiteness
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-20)
                C = B @ np.diag(D) @ B.T
                Cinv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

                x_mean = y_mean_clipped
                # Sigma adaptation
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

            return x_mean, best_f

        remaining = budget - n_calls
        if remaining > 20 and time_left() > 60:
            n_cma_steps = min(remaining - 10, 200)
            x_cma, f_cma = run_cma_es(best_x, best_f, n_cma_steps, sigma_init=0.5)

        # --- Phase 3: (1+1)-ES polishing from best point ---
        def run_1plus1_es(x0, f0, n_steps, rng_local, sigma_init=0.3):
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            fx = f0
            sigma = sigma_init
            for _ in range(n_steps):
                if n_calls >= budget or time_left() < 15:
                    break
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 10.0)
                else:
                    sigma = max(sigma * 0.85, 0.005)
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
            return x, fx

        remaining2 = budget - n_calls
        if remaining2 > 10 and time_left() > 30:
            n_es_steps = min(remaining2 - 5, 150)
            run_1plus1_es(best_x, best_f, n_es_steps, rng, sigma_init=0.2)

        # --- Phase 4: Final fine-tuning ---
        remaining3 = budget - n_calls
        if remaining3 > 5 and time_left() > 10:
            x = best_x.copy()
            fx = best_f
            sigma_tiny = 0.05
            for _ in range(remaining3):
                if time_left() < 5:
                    break
                y = np.clip(x + sigma_tiny * rng.standard_normal(dim), -5, 5)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_tiny = min(sigma_tiny * 1.1, 0.3)
                else:
                    sigma_tiny = max(sigma_tiny * 0.9, 0.001)
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx

        return best_x, best_f

# EVOLVE-BLOCK-END
