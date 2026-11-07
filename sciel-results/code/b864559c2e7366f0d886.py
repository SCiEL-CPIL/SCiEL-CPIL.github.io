"""Optimizer for BBOB-style f11 (Discus) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        d = self.dim
        budget = self.budget

        # Count function calls
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return func(x)

        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(d)
        fx = wrapped(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        # For discus function, anisotropic scaling is key
        # Start with a moderate sigma
        sigma = 1.0
        lam = 4  # population size
        # CMA-ES step sizes
        cc = (4 + d / lam) / ((d + 4) * (lam + 1.7))
        cs = (lam + 2 + cc) / ((d + lam + 5) * (lam + 1.7 - cc))
        c1 = 2 / ((d + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * (lam - 2 + 1 / lam) / ((d + 2) ** 2 + lam))

        # Mean
        m = x.copy()
        # Covariance matrix
        C = np.eye(d)
        # Evolution path for covariance
        pc = np.zeros(d)
        # Evolution path for step size
        ps = np.zeros(d)
        # Inverse square root of C (initially identity)
        Cinv_sqrt = np.eye(d)
        # Cholesky of C
        Cholesky = np.eye(d)
        # Eigen decomposition for updating C
        eigvals = np.ones(d)
        eigvecs = np.eye(d)
        # Damping for eigenvalue update
        deig = 1.0

        # Update counters
        i = 0  # generation counter

        while calls[0] < budget and time.time() < deadline:
            # Check time more frequently
            if time.time() > deadline - 5:
                break

            # Sample lambda offspring
            ys = []
            for j in range(lam):
                if calls[0] >= budget:
                    break
                # Generate normal random vector
                z = rng.standard_normal(d)
                # Transform using Cholesky
                y = m + sigma * (Cholesky @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = wrapped(y)
                ys.append((fy, y))

            if not ys:
                break

            # Sort by fitness
            ys.sort(key=lambda t: t[0])

            # Update best
            if ys[0][0] < best_f:
                best_f = ys[0][0]
                best_x = ys[0][1].copy()

            # Select mu best
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            # Weighted mean of selected solutions
            y_mean = np.zeros(d)
            for k in range(mu):
                y_mean += weights[k] * ys[k][1]

            # Update evolution path for step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ (y_mean - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (i + 1)) / np.sqrt(d) < 1.4 + 2 / (d + 1))

            # Update step size
            sigma *= np.exp(cs / d * (np.linalg.norm(ps) / np.sqrt(d) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path for covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_mean - m) / sigma

            # Update mean
            m = y_mean

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - (1 - delta_hsig) * C)
            for k in range(mu):
                yk = ys[k][1] - m
                C += cmu * weights[k] * np.outer(yk, yk)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2

            # Update eigen decomposition periodically
            if i % 10 == 0 or i == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip eigenvalues
                    eigvals = np.clip(eigvals, 1e-20, 1e10)
                    # Update Cholesky via eigen decomposition
                    # C = eigvecs @ diag(eigvals) @ eigvecs.T
                    # sqrt(C) = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
                    sqrt_eigvals = np.sqrt(eigvals)
                    Cholesky = eigvecs @ (sqrt_eigvals[:, None] * eigvecs.T)
                    # Inverse sqrt
                    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                    Cinv_sqrt = eigvecs @ (inv_sqrt_eigvals[:, None] * eigvecs.T)
                except np.linalg.LinAlgError:
                    # Fallback: reinitialize
                    C = np.eye(d)
                    Cholesky = np.eye(d)
                    Cinv_sqrt = np.eye(d)
                    eigvals = np.ones(d)
                    eigvecs = np.eye(d)

            i += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
