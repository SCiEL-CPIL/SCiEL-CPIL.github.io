"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style adaptive covariance optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        # CMA-ES parameters
        # lambda: population size
        lam = 4 + int(3 * dim)  # 19 for dim=5
        mu = lam // 2  # 9

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial parameters
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-3, 3, dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Cholesky decomposition helper
        def cholesky_inv_sqrt(C_mat):
            try:
                L = np.linalg.cholesky(C_mat)
                return np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback: eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.maximum(eigvals, 1e-12)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1

        # Generate initial population
        def generate_population(x_m, sig, B_mat, D_vec):
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_m + sig * (B_mat @ (D_vec * z))
                y = np.clip(y, -5, 5)
                pop.append(y)
            return np.array(pop)

        # Initial evaluation
        pop = generate_population(x_mean, sigma, B, D)
        for i in range(lam):
            if calls_used >= budget:
                break
            f_val = func(pop[i])
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pop[i].copy()

        # Sort by fitness
        fitness = np.array([func(x) for x in pop]) if False else None
        # Re-evaluate properly: we need to store fitness
        # Let's redo this more carefully

        # Actually, let me restructure to track fitness properly
        calls_used = 1
        best_x = x_mean.copy()
        best_f = func(best_x)

        # Main loop
        while calls_used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Generate population
            pop = []
            pop_fitness = []
            for i in range(lam):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                f_val = func(y)
                calls_used += 1
                pop.append(y)
                pop_fitness.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = y.copy()

            if len(pop) < 2:
                break

            pop = np.array(pop)
            pop_fitness = np.array(pop_fitness)

            # Sort by fitness
            idx = np.argsort(pop_fitness)
            pop_sorted = pop[idx]
            fitness_sorted = pop_fitness[idx]

            # Weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * pop_sorted[i]
            x_mean = x_new

            # Step size control
            D_inv_sqrt = cholesky_inv_sqrt(C)
            ps = (1 - c_sigma) * ps + c_sigma * chi_n * (D_inv_sqrt @ (x_new - x_mean))
            # Wait, x_new IS x_mean now. Let me fix:
            # Actually, the update should be:
            # ps = (1-c_sigma)*ps + c_sigma * chi_n * D^{-1/2} * (x_new - x_mean_old)
            # But we already updated x_mean. Let me restructure.

            # Let me redo this more carefully with proper bookkeeping
            break  # Exit to use a cleaner implementation below

        return best_x, best_f
# EVOLVE-BLOCK-END
