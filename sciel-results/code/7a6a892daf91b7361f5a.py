"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure CMA-ES optimizer with population size 10."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # CMA-ES parameters
        pop_size = 10  # lambda
        mu = pop_size // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / sum(weights**2)

        # Strategy parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial mean and step size
        # Start near center of box
        mean = np.zeros(dim)
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute sqrt of C for sampling
        # We'll use eigendecomposition each iteration for simplicity (dim=5 is small)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial evaluation of the mean
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        calls_made += 1
        update_best(x0, f0)

        # CMA-ES main loop
        while calls_made < budget - 1 and time_ok():
            if calls_made >= budget - 5:
                break

            # Sample population
            # Get sqrt of C via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip negative eigenvalues
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Sample
            samples = []
            for i in range(pop_size):
                z = np.random.standard_normal(dim)
                x = mean + sigma * (sqrt_C @ z)
                x = np.clip(x, -5, 5)
                samples.append(x)

            # Evaluate
            fvals = []
            for x in samples:
                if calls_made >= budget - 1 or not time_ok():
                    break
                f = func(x)
                calls_made += 1
                fvals.append(f)
                update_best(x, f)

            if len(fvals) < pop_size:
                # Pad with the best found so far to maintain population size
                while len(fvals) < pop_size:
                    fvals.append(best_f)
                    samples.append(best_x.copy() if best_x is not None else np.zeros(dim))

            # Sort by fitness
            order = np.argsort(fvals)
            samples = np.array(samples[order])
            fvals = np.array(fvals[order])

            # Update mean
            old_mean = mean.copy()
            mean_new = sum(weights[i] * samples[i] for i in range(mu))

            # Update evolution paths
            diff = (mean_new - old_mean) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * diff
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (sqrt_C.T @ diff)

            # hsig
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim) + 1 / (12 * dim * dim * dim))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // pop_size))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update covariance
            rank_one = np.outer(pc, pc)
            if hsig:
                rank_mu = sum(weights[i] * np.outer(samples[i] - old_mean, samples[i] - old_mean) for i in range(mu))
            else:
                rank_mu = np.zeros((dim, dim))

            C = (1 + c1 + cmu - c1) * C + c1 * (rank_one - (1 - cc) * (1 - 2 * cc) * C) + cmu * rank_mu

            # Ensure C is positive semi-definite
            # Symmetrize
            C = (C + C.T) / 2
            eigvals_c, eigvecs_c = np.linalg.eigh(C)
            eigvals_c = np.maximum(eigvals_c, 1e-20)
            C = eigvecs_c @ np.diag(eigvals_c) @ eigvecs_c.T

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Convergence check
            if np.linalg.norm(mean_new - old_mean) < 1e-10 * sigma:
                break

            mean = mean_new

        # Final polish with a few random perturbations around best
        if best_x is not None and calls_made < budget - 1 and time_ok():
            x = best_x.copy()
            fx = best_f
            rng = np.random.default_rng(42)
            for _ in range(min(10, budget - calls_made - 1)):
                if not time_ok():
                    break
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                y = np.clip(x + 0.05 * direction, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(best_x)
                calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
