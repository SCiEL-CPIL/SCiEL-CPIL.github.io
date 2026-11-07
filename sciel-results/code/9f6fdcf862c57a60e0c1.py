"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: random exploration + CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Random exploration to find a good starting point
        n_explore = min(30, budget // 5)
        best_x = None
        best_f = np.inf

        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best point
        remaining = budget - n_explore
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 4 * int(1 + 3 * dim)  # ~48 for dim=5
        pop_size = min(pop_size, max(4, remaining - 1))
        mu = pop_size // 2

        # CMA-ES state
        x_mean = best_x.copy()
        f_mean = best_f
        sigma = 1.0  # start with moderate step
        C = np.eye(dim)
        # Eigen-decompose for C
        # Use simplified update: C = I initially, update via rank-1 and rank-mu
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / (pop_size - 2)) / ((pop_size + 2) + 4 + dim / (pop_size - 2))
        cs = (2 + dim / (pop_size - 2)) / (pop_size + 2 + dim / (pop_size - 2))
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(1 - c1, 2 * ((mu + 0.5) - 2) / ((dim + 2) ** 2 + pop_size))
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1) - 1) + cs

        # Precompute recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        evals_used = n_explore

        for gen in range(remaining):
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((pop_size, dim))
            # Apply C via Cholesky or eigendecomposition
            # For simplicity in 5D, use eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.sqrt(np.maximum(eigvals, 1e-12))
            # z_i = B_i @ eigvecs * eigvals
            z = (B * eigvals) @ eigvecs.T
            # Actually, we want: y = x_mean + sigma * z where z ~ N(0, C)
            # z = eigvecs @ diag(sqrt(eigvals)) @ B.T ... let me redo
            # B is (pop_size, dim), we want z_i = eigvecs @ diag(sqrt(eigvals)) @ B_i
            # So z = B @ (eigvecs * eigvals).T  -- no
            # z_i = sum_j B_ij * eigvecs[j,k] * eigvals[k]
            # z = B @ (eigvecs * eigvals[None, :]).T  -- this is (pop, dim) @ (dim, dim)
            # Actually: z = B @ (eigvecs * eigvals).T
            z = B @ (eigvecs * eigvals).T
            pop = x_mean + sigma * z
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.empty(pop_size)
            for i in range(pop_size):
                if time.time() > deadline:
                    # Fill remaining with large values
                    fvals[i:] = np.inf
                    break
                fvals[i] = func(pop[i])
                evals_used += 1

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = pop[order]
            f_sorted = fvals[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Update mean
            x_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
                C @ (x_new - x_mean) / sigma
            )

            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1))) < \
                   (1 + 2 / (dim + 1)) * np.sqrt(dim + 1)
            if hsig:
                pc = (1 - cc) * pc + cc * (x_new - x_mean) / sigma

            # Update C
            delta_hsig = (1.0 if hsig else 0.0) - (1 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu / sigma ** 2 * np.sum(
                    weights[:, None, None] *
                    np.outer(y_sorted[:mu] - x_mean, y_sorted[:mu] - x_mean),
                    axis=0
                )

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Re-orthogonalize via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            x_mean = x_new

            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
