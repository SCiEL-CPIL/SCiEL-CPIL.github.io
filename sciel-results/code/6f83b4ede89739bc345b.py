"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB-style f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Use a fixed seed for reproducibility within a single call
        rng = np.random.default_rng(42)

        # Initialize CMA-ES
        # Start from origin (0,0,...,0) which is often good for Rosenbrock
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # CMA-ES parameters
        sigma = 2.0  # Initial step size
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
        # Means and covariances
        m = x.copy()
        C = np.eye(dim)
        # Step sizes for C and sigma
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) ** 2 + 4)
        cs = (dim + 1.7) / (dim + 9)
        cmu = min(2 - cc + 2 / (dim - 2), 0.5 / (dim + 1.3))
        alpha_sigma = (1 + 2 * max(0, np.sqrt((dim + 1) / (dim + 4) - 1)) + cs)
        # Eigenvalues of C
        damps = 1 + 2 * max(0, np.sqrt((dim + 1) / (dim + 4) - 1)) + cs
        # Covariance matrix adaptation step size
        damp_c = 1 + 2 * max(0, np.sqrt((dim + 1) / (dim + 4) - 1)) + cs

        # Precompute Cholesky factor
        C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
        # Normal distribution mean for z
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21. * dim ** 2))

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Main loop
        while calls_made < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Sample population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = np.clip(m + sigma * (C_sqrt_inv.T @ z), -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls_made += 1
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()

            # Weighted mean
            mu = lambda_ // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1 / np.sum(w ** 2)
            m_new = np.sum(w[:, None] * Y[:mu], axis=0)

            # Update evolution paths
            C_sqrt = np.linalg.cholesky(C)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv.T @ (m_new - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - lambda_)))) / chi_n < (1 + 2 / (dim + 1)) * 1.4
            if hsig:
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - m) / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cs * damp_c + cmu - 1) * C
            C += (1 - cc) * (np.outer(pc, pc) - C) * (1 - (1 - cc) ** 2)  # This is an approximation, proper update is complex
            # Proper CMA-ES covariance update:
            C = (1 + cs * damp_c + cmu - 1) * C
            C += (1 - cc) * (np.outer(pc, pc) - C)  # Approximation
            rank_mu = np.sum(w[:, None, None] * np.outer(Y[:mu] - m, Y[:mu] - m), axis=0)
            C += cmu * (rank_mu - C)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damp_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            m = m_new
            x = m.copy()

            # Re-clip m to box (CMA-ES mean should stay in bounds ideally, but we clip samples)
            m = np.clip(m, -5, 5)

            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset covariance to identity
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sigma = 1.0

            # If we used up budget in the inner loop, break
            if calls_made >= budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
