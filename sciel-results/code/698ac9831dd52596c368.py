"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for minimizing func on [-5, 5]^dim with hidden shift/rotation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # Safety margin before 330s kill

        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of weighted individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # CMA-ES adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-3, 3, dim)
        sigma = 1.0

        # Covariance matrix C = I, and its Cholesky factor
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Track for early stopping
        no_improve = 0
        patience = 50

        while calls < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Sample population
            # Use Cholesky factorization for sampling from N(x, sigma^2 * C)
            # Update Cholesky periodically (every 10 generations)
            if calls % 10 == 0:
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

            # Generate population
            z = rng.standard_normal((lam, dim))
            pop = np.array([x + sigma * z[i] @ L.T for i in range(lam)])
            # Clip to box
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.array([func(pop[i]) for i in range(lam)])
            calls += lam

            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_fx:
                best_fx = fvals[idx_best]
                best_x = pop[idx_best].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping if no improvement for many generations
            if no_improve > patience:
                # Try a restart with larger sigma
                x = best_x.copy()
                sigma = max(sigma * 2.0, 0.5)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve = 0
                if calls >= budget:
                    break

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = pop[order]
            f_sorted = fvals[order]

            # Weighted mean
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update covariance matrix adaptation
            y_vec = (y_w - x) / sigma
            pc = (1 - c_c) * pc + c_c * y_vec
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

            # Update C
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) / np.sqrt(dim)
            if hsig < 1.4 + 2 / (dim + 1):
                C = (1 + c_1 + c_mu - c_1) * C
                C += c_1 * (np.outer(pc, pc) - C)
                rank_mu = np.sum(weights[:, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x) / sigma**2, axis=0)
                C += c_mu * (rank_mu - C)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C = C + 1e-8 * np.eye(dim)
            except:
                C = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = y_w
            x = np.clip(x, -5, 5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
