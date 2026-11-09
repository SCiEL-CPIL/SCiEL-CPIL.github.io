"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        # CMA-ES Parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # Standard choice
        mu = lambda_ // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)  # Effective weight
        c1 = (2 / ((dim + 1)**2 + 10)) / 2  # Covariance adaptation rate
        cmu = min(1 - c1, 2 * ((mueff + 2) / (dim + 2))**2 - 1) / 2
        cs = (mueff + 1.7) / (dim + mueff + 5)  # Step size control rate
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = 4 / (dim + 4)  # Mean shift rate
        c1 = (2 / ((dim + 1.7)**2 + 10))  # Corrected c1
        cmu = min(1 - c1, 2 * ((mueff + 2) / (dim + 2))**2 - 1)

        # Initialization
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility within instance
        x_mean = np.zeros(dim)  # Start at center, as minimizer is in [-4,4]^5
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Best found
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Precompute constants
        inv_sqrt_det_C = 1.0

        for gen in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            Y = rng.standard_normal((lambda_, dim))
            # Transform Y using Cholesky decomposition of C
            # We maintain C_sqrt_inv for sampling? No, standard CMA-ES samples Z ~ N(0, C)
            # Efficiently: X = mean + sigma * (C_sqrt @ Z)
            # We can update C_sqrt via Cholesky every generation or use eigendecomp.
            # For dim=5, Cholesky is very fast.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite due to numerical issues
                C = np.eye(dim)
                L = np.eye(dim)
            
            X = x_mean + sigma * (Y @ L.T)
            
            # Evaluate population
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                F[i] = func(X[i])
                calls_made += 1
            
            # Sort by fitness
            sorted_indices = np.argsort(F)
            X_sorted = X[sorted_indices]
            F_sorted = F[sorted_indices]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Compute weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * X_sorted[i]
            
            # Step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (x_new - x_mean) / sigma
            # Note: C_sqrt_inv is actually C^{-1/2}. We need to compute it.
            # Let's compute C_inv_sqrt from C.
            # For dim=5, we can just use eigendecomp or solve.
            # Actually, the standard formula uses C^{-1/2} @ (x_new - x_mean).
            # Let's compute C_inv_sqrt properly.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues for stability
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            # Recalculate ps with correct C_inv_sqrt
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (x_new - x_mean)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Mean update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new - x_mean) / sigma
            x_mean = x_mean + cc * pc

            # Covariance update
            delta_sig = np.linalg.norm(ps) / chi_n - 1
            sigma *= np.exp(cs / damps * (delta_sig - 1))
            
            # Rank-1 update
            C = (1 - c1 - cmu) * C
            C += (c1 + hsig * c1 * (2 - cc)) * np.outer(pc, pc)
            
            # Rank-mu update
            if cmu > 0:
                # Y_sorted is the normalized steps
                Y_diff = (X_sorted[:mu] - x_mean) / sigma
                # Weighted outer sum
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(Y_diff[i], Y_diff[i])
                C += cmu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if necessary
            eigvals_check, _ = np.linalg.eigh(C)
            if np.any(eigvals_check < 1e-10):
                C += 1e-10 * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
