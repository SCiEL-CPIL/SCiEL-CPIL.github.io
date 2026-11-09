"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize population-based CMA-ES
        # lambda: number of offspring per generation
        lam = max(4, dim + 1)  # small population for dim=5
        mu = lam // 2  # number of parents

        # Weights for recombination
        weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) / 3
        cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((dim + 2) ** 2 + mueff))
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: use a few random starts to find a good basin
        # For Rastrigin, many local minima exist. Let's start with a decent initialization.
        # Try a few initial points and pick the best one.
        x0_candidates = []
        for _ in range(min(10, budget // 50)):
            x_cand = rng.uniform(-5, 5, dim)
            x0_candidates.append((func(x_cand), x_cand))
        x0_candidates.sort(key=lambda t: t[0])
        x = x0_candidates[0][1].copy()
        fx = x0_candidates[0][0]

        # Initial sigma
        sigma = 1.0

        # Covariance matrix
        C = np.eye(dim)
        # Evolve C
        C_sqrt_inv = np.eye(dim)

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        hsig = 0.0

        # Track best solution
        best_x = x.copy()
        best_f = fx

        calls_made = 10  # we used 10 calls for initialization
        remaining = budget - calls_made

        if remaining <= 0:
            return best_x, best_f

        while calls_made < budget:
            # Determine how many offspring to generate in this generation
            # We want to use remaining budget efficiently
            if remaining < lam:
                lam_gen = remaining
            else:
                lam_gen = lam

            # Generate offspring
            offspring = []
            for i in range(lam_gen):
                # Sample from multivariate normal with current C and sigma
                # Use Cholesky or just standard normal scaled by sigma * C_sqrt_inv
                # For simplicity, use standard normal * sigma (isotropic) initially,
                # but let's implement proper CMA-ES
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Select top mu offspring
            selected = offspring[:min(mu, len(offspring))]

            # Weighted mean of selected
            new_x = np.zeros(dim)
            w_sum = 0
            for i, (f_sel, y_sel) in enumerate(selected):
                w = weights[i]
                new_x += w * y_sel
                w_sum += w
            new_x /= w_sum

            # Update covariance matrix
            diff = new_x - x
            diff = diff / sigma  # normalize

            # Rank-one update
            C += c1 * (np.outer(diff, diff) - C)
            # Rank-mu update
            for i, (f_sel, y_sel) in enumerate(selected):
                d = (y_sel - x) / sigma
                C += cmu * weights[i] * (np.outer(d, d) - C)

            # Ensure positive semi-definite by eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T

            # Update C_sqrt_inv
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_sqrt_inv = (eigvecs * eigvals_inv_sqrt) @ eigvecs.T

            # Update step size
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * diff
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update p_c
            hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - 10))) < (1.4 + 2 / (dim + 1)) * chi else 0.0
            p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff

            x = new_x
            remaining = budget - calls_made

        return best_x, best_f
# EVOLVE-BLOCK-END
