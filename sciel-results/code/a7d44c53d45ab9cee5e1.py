"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        N = self.dim
        lambda_ = 4 * N  # population size
        mu = lambda_ // 2  # number of parents

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (N + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
        c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        c_1 = 2 / (N**1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (N**1.8 + 10))

        # Initial state
        x_mean = np.zeros(N)
        sigma = 2.0
        B = np.eye(N)
        D = np.ones(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))

        best_x = x_mean.copy()
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            return f

        try:
            # Initial population
            for _ in range(100):  # Check if we can even start
                if time.time() > deadline:
                    break

            # Generate initial population
            pop = x_mean + sigma * B @ (D * np.random.default_rng(0).standard_normal((lambda_, N)))
            pop = np.clip(pop, -5, 5)
            fits = [safe_func(x) for x in pop]

            # Sort
            idx = np.argsort(fits)
            pop = pop[idx]
            fits = np.array(fits)[idx]

            # Update best
            if fits[0] < best_f:
                best_f = fits[0]
                best_x = pop[0].copy()

            # Main loop
            while call_count < self.budget - 10 and time.time() < deadline:
                # Update mean
                w = weights
                x_mean_new = np.sum(w[:, None] * pop[:mu], axis=0)

                # Update step size
                y_mean = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (N + 1)) * chiN:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))

                # Update covariance
                pc = (1 - c_c) * pc + c_c * (x_mean_new - x_mean) / sigma
                C = B @ np.diag(D**2) @ B.T
                C += (c_1 + c_mu) * (np.outer(y_mean, y_mean) - C) + c_1 * C
                # Actually, standard update is:
                # C = (1 - c_1 - c_mu) * C + c_1 * (C + pc*pc.T) + c_mu * sum(wi * yi*yi.T)
                # Let's use the simpler rank-1 and rank-mu update on the matrix directly
                # But we need to decompose C for B and D.
                
                # Recompute C properly
                C = (1 - c_1 - c_mu) * C + c_1 * (C + np.outer(pc, pc)) 
                # Wait, the standard update is:
                # C = (1 - c1 - cmu) C + c1 (C + pc pc^T) + cmu sum wi yi yi^T
                # Let's stick to updating the matrix C and decomposing it.
                
                # Correct C update:
                C = (1 - c_1 - c_mu) * C + c_1 * (C + np.outer(pc, pc)) + c_mu * np.sum(
                    w[:, None, None] * np.einsum('i,ij,ik->ijk', np.ones(mu), pop[:mu] - x_mean, pop[:mu] - x_mean) / sigma**2, axis=0
                )
                
                # Ensure symmetry and positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    B = np.eye(N)
                    D = np.ones(N)

                x_mean = x_mean_new

                # Generate next population
                pop = x_mean + sigma * B @ (D * np.random.default_rng(0).standard_normal((lambda_, N)))
                pop = np.clip(pop, -5, 5)
                fits = [safe_func(x) for x in pop]

                # Sort
                idx = np.argsort(fits)
                pop = pop[idx]
                fits = np.array(fits)[idx]

                # Update best
                if fits[0] < best_f:
                    best_f = fits[0]
                    best_x = pop[0].copy()

        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
