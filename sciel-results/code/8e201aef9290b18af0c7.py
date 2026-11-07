"""Sobol global phase + CMA-ES local refinement for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random global search followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Phase 1: Sobol quasi-random global exploration
        # Use scipy's Sobol sequence for low-discrepancy sampling
        from scipy.stats import qmc

        # Allocate ~40% of budget to global phase
        n_global = min(budget // 2, 200)

        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        # Number of points: power of 2
        m = 1
        while m < n_global:
            m *= 2
        # Generate Sobol points in [0,1]^n
        sobol_points = sampler.random(m)
        # Map to [-5, 5]^n
        sobol_points = sobol_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        call_count = 0

        for i in range(m):
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            x = sobol_points[i]
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES local refinement around the best point
        if call_count < budget and best_x is not None:
            remaining = budget - call_count
            if remaining > 5:
                best_x, best_f = self._cmaes(func, best_x, best_f, remaining, n, call_count, deadline)
                call_count += remaining  # approximate

        return best_x, best_f

    def _cmaes(self, func, x0, f0, budget, dim, calls_made, deadline):
        """Simple CMA-ES implementation for local refinement."""
        # Initialize CMA-ES parameters
        sigma = 1.0  # initial step size
        # Covariance matrix C (identity initially)
        C = np.eye(dim)
        # Inverse of C
        C_inv = np.eye(dim)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.sqrt(np.maximum(eigvals, 1e-20))

        # Adaptation parameters
        cs = (dim + 2) ** -0.5 - 1 / (4 * dim)
        c_sigma = (5 + dim / 2) / (dim + 5 + 2 * dim / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + dim)
        cmu = min(1 - c1, 2 * ((dim - 2 + 2. / dim) ** 2 - 1) / ((dim + 2) ** 2 + dim))
        mu = dim // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1. / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        mean = x0.copy()
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        hsig = 0.0
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        count_steps = 0

        lambda_pop = min(dim + 3, budget // 2)  # population size
        lambda_pop = max(lambda_pop, 4)

        for gen in range(budget // lambda_pop):
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Generate offspring
            offspring = []
            offspring_f = []

            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Sample from N(mean, sigma^2 * C)
                z = np.random.standard_normal(dim)
                # Transform using eigen decomposition
                y = eigvecs @ (eigvals * z)
                x_new = mean + sigma * y
                x_new = np.clip(x_new, -5, 5)

                fx = func(x_new)
                calls_made += 1
                offspring.append(x_new)
                offspring_f.append(fx)

            if len(offspring) < 2:
                break

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]

            # Update best
            if offspring_f[0] < f0:
                f0 = offspring_f[0]
                x0 = offspring[0].copy()

            # Recombination
            y_w = np.zeros(dim)
            for i in range(min(mu, len(offspring))):
                y_w += weights[i] * (offspring[i] - mean) / sigma

            # Update mean
            mean = mean + c1 * (x0 - mean) + cmu * (y_w - (c1 / cmu) * (mean - x0)) if cmu > 0 else mean + c1 * (x0 - mean)
            # Simpler: just move toward best
            mean = 0.5 * mean + 0.5 * x0

            # Update evolution paths
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim)
            ps = (1 - c_sigma) * ps + c_sigma_sqrt * (1 if f0 < offspring_f[-1] else 0) * np.linalg.norm(y_w) / (sigma * chi_n) if False else (1 - c_sigma) * ps + c_sigma_sqrt * y_w / sigma
            ps_norm = np.linalg.norm(ps)
            hsig = ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1) / damps)) / chi_n < (1.4 + 2 / (dim + 1)) * (1 - c_sigma)

            pc = (1 - cs) * pc + float(hsig) * cs * y_w / sigma
            # Avoid numerical issues
            if np.linalg.norm(pc) > (1.7 + 1 / cs) * chi_n:
                C_inv = C_inv  # reset
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.sqrt(np.maximum(eigvals, 1e-20))

            # Update covariance
            delta_diag = (1 + 2 * float(hsig - 1) * cs) * cs
            C = (1 + c1 * delta_diag - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for i in range(min(mu, len(offspring))):
                y_i = (offspring[i] - mean) / sigma
                C += cmu * weights[i] * np.outer(y_i, y_i)

            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            try:
                eigvals_c, eigvecs_c = np.linalg.eigh(C)
                eigvals_c = np.maximum(eigvals_c, 1e-10)
                C = (eigvecs_c * eigvals_c) @ eigvecs_c.T
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = eigvals_c ** 0.5, eigvecs_c
            except:
                C = np.eye(dim)
                eigvals, eigvecs = np.ones(dim), np.eye(dim)

            # Adapt sigma
            sigma *= np.exp(c_sigma / damps * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Clip mean to bounds
            mean = np.clip(mean, -5, 5)

        return x0, f0
# EVOLVE-BLOCK-END
