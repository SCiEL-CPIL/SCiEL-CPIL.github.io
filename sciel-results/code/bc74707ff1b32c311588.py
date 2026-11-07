"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim + 1))
        lam = int(4 + 3 * np.log(dim + 1))
        # For dim=5: lam = 4 + 3*floor(2.0) = 10, but let's use a larger pop for multimodality
        # Actually standard CMA-ES uses lam = 4 + floor(3*log(n+1))
        # For n=5: log(6)=1.79, 3*1.79=5.37, floor=5, lam=9
        # Let's use lam=20 for better exploration in multimodal landscape
        lam = 20

        # Weighting
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        invsqrtC = np.eye(dim)
        # Cholesky of C: C = C * C.T, we need C such that C @ C.T = covariance
        # Actually C is the covariance matrix, and we factor it as C = B @ B.T
        # For sampling: x + sigma * B @ z where z ~ N(0, I)
        # Let's maintain B (Cholesky factor)
        B = np.eye(dim)

        # Count function evaluations
        nfeval = 0
        best_x = x.copy()
        best_f = np.inf

        # Helper to update Cholesky
        def update_cholesky(C_mat):
            nonlocal B
            try:
                B = np.linalg.cholesky(C_mat)
            except np.linalg.LinAlgError:
                # Add small regularization
                C_mat = C_mat + 1e-10 * np.eye(dim)
                B = np.linalg.cholesky(C_mat)

        # Initial sampling
        # We'll do a random restart approach: start with a few random points
        # to get a good initial population, then run CMA-ES

        # Actually, let's just start CMA-ES from origin with sigma=2
        # and let it adapt. But for multimodal f21, we might want to
        # do a few random evaluations first to find a good region.

        # Let's do a hybrid: first 20 evaluations are random to find
        # a promising region, then switch to CMA-ES centered on best point.

        # Phase 1: Random exploration (20 evaluations)
        rng = np.random.default_rng(42)
        for i in range(min(20, budget)):
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            nfeval += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

            if nfeval >= budget or time.time() > deadline:
                break

        if nfeval >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES from best point
        x = best_x.copy()
        sigma = 1.0  # smaller sigma for local search
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        update_cholesky(C)

        # Main CMA-ES loop
        while nfeval < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pop_x = []
            pop_f = []

            for i in range(lam):
                if nfeval >= budget:
                    break
                if time.time() > deadline:
                    break

                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                f_y = func(y)
                nfeval += 1
                pop_x.append(y)
                pop_f.append(f_y)

                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()

            if len(pop_x) < 2:
                break

            pop_x = np.array(pop_x)
            pop_f = np.array(pop_f)

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()

            # Compute weighted mean
            mu_x = np.sum(weights[:, np.newaxis] * pop_x[:mu], axis=0)

            # Update step size control
            y_vec = (mu_x - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_vec
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (24 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_vec

            # Update covariance matrix
            delta = 0 if (norm_ps**2 / (1 - (1 - c_sigma)**(2 * (nfeval + 1)))) < (2 + 2 / mueff) * 1.1 else 1
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(pop_x[:mu] - x, pop_x[:mu] - x) / sigma**2, axis=0)
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            update_cholesky(C)

            # Update mean
            x = mu_x

            # Stopping criterion: convergence
            if sigma < 1e-10 or np.all(np.abs(sigma * B) < 1e-10):
                break

            # If no improvement for a while, reduce sigma or restart
            # Simple: if sigma is very small, break
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
