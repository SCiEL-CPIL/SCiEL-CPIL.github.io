"""CMA-ES optimizer for BBOB-style Schaffers F7 (f17)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance, population size ~4+3*dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        # CMA-ES parameters
        pop_size = int(np.ceil(4 + 3 * dim))  # 19 for dim=5
        # Initial mean: center of the box
        x_mean = np.zeros(dim)
        # Initial sigma: explore broadly
        sigma = 3.0
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Eigen decomposition cache
        B = np.eye(dim)
        D = np.ones(dim)
        # Step sizes
        cs = (pop_size + 2) / (dim + pop_size + 5)
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
        cc = (4 - dim / pop_size + cs) / (dim + 4 - cs)
        cmu = min(2 / ((dim + 1.3) ** 2 + pop_size), 0.5 - cs / 2)
        # Weights (logarithmic)
        weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        # Damping
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x_mean.copy()
        best_f = float('inf')

        # Initial evaluation of center
        x0 = np.clip(x_mean, -5, 5)
        f0 = func(x0)
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        calls_made = 1
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        while calls_made < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Update eigen decomposition (recompute every iteration for simplicity at dim=5)
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)

            # Generate population
            ys = []
            xs = []
            for i in range(pop_size):
                z = np.random.standard_normal(dim)
                # Transform to C-space
                y = B @ (D * z)
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                xs.append(x)
                ys.append(y)

            # Evaluate population
            fs = []
            for x in xs:
                if time.time() > deadline:
                    break
                f = func(x)
                fs.append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            if len(fs) < pop_size:
                break

            fs = np.array(fs)
            xs = np.array(xs)
            ys = np.array(ys)

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]

            # Update mean
            x_mean_new = np.sum(weights[:, None] * xs_sorted, axis=0)
            x_mean_new = np.clip(x_mean_new, -5, 5)

            # Update path for sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D * (B.T @ (x_mean_new - x_mean) / sigma)))
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - pop_size) // pop_size))) / chiN - 1
            if hsig < 0.8:
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))

            # Update path for C
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * (B.T @ (D * (B.T @ (x_mean_new - x_mean) / sigma)))
            hc = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** (2 * (calls_made - pop_size) // pop_size))) / chiN - 1

            # Update covariance
            delta_hc = (1 + hc) * cc * (2 - cc)
            C = (1 - cs - cmu + delta_hc) * C + cmu * np.sum(
                weights[:, None, None] * np.outer(ys_sorted, ys_sorted), axis=0
            ) + delta_hc * np.outer(p_c, p_c)

            # Ensure positive definite
            C = (C + C.T) / 2
            # Regularize
            C += 1e-10 * np.eye(dim)

            x_mean = x_mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
